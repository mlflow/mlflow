import { TraceInfo } from '../core/entities/trace_info';
import { Trace } from '../core/entities/trace';
import {
  CreateAssessment,
  CreateExperiment,
  CreateIssue,
  CreateTraceInfoV4,
  DatabricksIssues,
  DeleteExperiment,
  ExportOtlpTraces,
  ExportOtlpTracesOss,
  GetExperiment,
  GetExperimentByName,
  GetIssue,
  GetTraceInfoV3,
  SearchTracesV3,
  SearchIssues,
  StartTraceV3,
  UpdateIssue,
} from './spec';
import {
  AssessmentSourceType,
  assertFeedbackPayload,
  assertMetadataStringMap,
  Feedback,
  standardizeSourceType,
  type AssessmentError,
  type AssessmentSource,
  type FeedbackValueType,
} from '../core/entities/assessment';
import { makeRawRequest, makeRequest, MlflowHttpError } from './utils';
import { TraceData } from '../core/entities/trace_data';
import { ArtifactsClient, getArtifactsClient } from './artifacts';
import { AuthProvider, HeadersProvider, isDatabricksUri } from '../auth';
import {
  Issue,
  IssueSeverity,
  IssueStatus,
  type IssueSeverityName,
  type IssueStatusName,
} from '../core/entities/issue';
import { DATABRICKS_UC_TABLE_HEADER, MLFLOW_EXPERIMENT_ID_HEADER } from '../core/constants';
import { serializeTraceLocation, type TraceLocation } from '../core/entities/trace_location';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-proto';
import type { ReadableSpan as OTelReadableSpan } from '@opentelemetry/sdk-trace-base';
import { ExportResultCode, type ExportResult } from '@opentelemetry/core';

export interface SearchTracesOptions {
  /** Trace locations (e.g. MLflow experiments) to search over. */
  locations: TraceLocation[];
  /** Search filter string, e.g. `"tags.env = 'prod'"`. */
  filter?: string;
  /** Maximum number of traces to return in a single page. */
  maxResults?: number;
  /** List of order-by clauses, e.g. `['timestamp_ms DESC']`. */
  orderBy?: string[];
  /** Pagination token obtained from a previous searchTraces call. */
  pageToken?: string;
  /**
   * Whether to fetch span data for each matching trace. Defaults to `true`.
   * When `false`, only trace metadata is returned and the traces have empty
   * span data.
   */
  includeSpans?: boolean;
}

export interface SearchTracesResult {
  traces: Trace[];
  nextPageToken?: string;
}

export interface CreateIssueOptions {
  experimentId: string;
  name: string;
  description: string;
  severity: IssueSeverityName;
  status?: IssueStatusName;
  categories?: string[];
  metadata?: Record<string, string>;
  issueId?: string;
  /** Most recent observation in milliseconds since the epoch. */
  lastSeenTimestamp?: number;
  sourceRunId?: string;
}

export interface UpdateIssueOptions {
  experimentId: string;
  name?: string;
  /** An empty string clears the description. */
  description?: string;
  severity?: IssueSeverityName;
  status?: IssueStatusName;
  /** Replaces the categories; an empty array clears them. */
  categories?: string[];
  /** Metadata entries to add or replace. */
  metadata?: Record<string, string>;
  deleteMetadataKeys?: string[];
  lastSeenTimestamp?: number;
}

export interface SearchIssuesOptions {
  experimentId: string;
  /** Status comparisons (=, !=) joined by AND; OSS also supports source_run_id. */
  filter?: string;
  maxResults?: number;
  /** Backend field names, e.g. ['created_timestamp DESC']. Databricks supports one field. */
  orderBy?: string[];
  pageToken?: string;
  /** Defaults to full. Basic omits the description. */
  view?: 'basic' | 'full';
}

export interface SearchIssuesResult {
  issues: Issue[];
  nextPageToken?: string;
  totalCount?: number;
}

/**
 * Client for MLflow tracing operations.
 *
 * Supports multiple authentication methods:
 * - Databricks: PAT tokens, OAuth M2M, Azure CLI/MSI, Google Cloud
 * - OSS MLflow: Basic auth, Bearer tokens, or no auth
 */
export class MlflowClient {
  /** Client implementation to upload/download trace data artifacts */
  private artifactsClient: ArtifactsClient;

  /** Headers provider for authenticated requests */
  private headersProvider: HeadersProvider;

  /** Host URL for API requests */
  private hostUrl: string;

  private isDatabricks: boolean;

  /**
   * Creates a new MlflowClient.
   *
   * @param trackingUri - The tracking URI (e.g., "databricks", "http://localhost:5000")
   * @param authProvider - The authentication provider to get tokens for authenticated requests
   */
  constructor(options: { trackingUri: string; authProvider: AuthProvider }) {
    this.headersProvider = options.authProvider.getHeadersProvider();
    this.hostUrl = options.authProvider.getHost();
    this.isDatabricks = isDatabricksUri(options.trackingUri);

    this.artifactsClient = getArtifactsClient({
      trackingUri: options.trackingUri,
      host: this.hostUrl,
      authProvider: options.authProvider,
    });
  }

  /**
   * Get the host URL for this client.
   */
  getHost(): string {
    return this.hostUrl;
  }

  // === TRACE LOGGING METHODS ===
  /**
   * Create a new TraceInfo record in the backend store.
   * Corresponding to the Python SDK's start_trace_v3() method.
   *
   * Note: the backend API is named as "Start" due to unfortunate miscommunication.
   * The API is indeed called at the "end" of a trace, not the "start".
   */
  async createTrace(traceInfo: TraceInfo): Promise<TraceInfo> {
    const url = StartTraceV3.getEndpoint(this.hostUrl);
    const payload: StartTraceV3.Request = { trace: { trace_info: traceInfo.toJson() } };
    const response = await makeRequest<StartTraceV3.Response>(
      'POST',
      url,
      this.headersProvider,
      payload,
    );
    return TraceInfo.fromJson(response.trace.trace_info);
  }

  /**
   * Create a new TraceInfo record in the Databricks V4 (Unity Catalog)
   * trace-info endpoint. Used for UC-backed traces so that trace-level
   * tags and metadata persist alongside spans uploaded via OTLP.
   *
   * Corresponds to Python's `DatabricksRestStore._start_trace_v4`.
   *
   * @param location - The UC location string ("catalog.schema" or
   *                   "catalog.schema.table_prefix") that comes from
   *                   the trace ID `trace:/<location>/<hex>`.
   * @param otelTraceId - The hex OTel trace ID portion.
   * @param traceInfo - The full TraceInfo to persist; the backend reads
   *                    `tags`, `trace_metadata`, `state`, durations, etc.
   */
  async createTraceInfoV4(
    location: string,
    otelTraceId: string,
    traceInfo: TraceInfo,
  ): Promise<TraceInfo> {
    const url = CreateTraceInfoV4.getEndpoint(this.hostUrl, location, otelTraceId);
    const payload: CreateTraceInfoV4.Request = traceInfo.toJson();
    const response = await makeRequest<CreateTraceInfoV4.Response>(
      'POST',
      url,
      this.headersProvider,
      payload,
    );
    return TraceInfo.fromJson(response);
  }

  /**
   * Upload OTel spans to a Databricks Unity Catalog location via the OTLP
   * HTTP+protobuf endpoint. The `spansTableName` is the fully qualified
   * spans table (catalog.schema.table) and is forwarded as the
   * `X-Databricks-UC-Table-Name` header used by Databricks to route the
   * payload to the correct UC location.
   *
   * Uses `@opentelemetry/exporter-trace-otlp-proto` which serializes the
   * spans to the OTLP `ExportTraceServiceRequest` protobuf wire format —
   * the Databricks endpoint only accepts `application/x-protobuf`, not
   * the OTLP/HTTP+JSON form.
   */
  async exportOtlpSpansToUc(spans: OTelReadableSpan[], spansTableName: string): Promise<void> {
    if (spans.length === 0) {
      return;
    }
    // Fetch fresh auth headers for every export so OAuth-rotated tokens are
    // honored. The exporter takes a static headers map at construction time,
    // so we rebuild it per call rather than caching an instance.
    const authHeaders = await this.headersProvider();
    const exporter = new OTLPTraceExporter({
      url: ExportOtlpTraces.getEndpoint(this.hostUrl),
      headers: {
        ...authHeaders,
        [DATABRICKS_UC_TABLE_HEADER]: spansTableName,
      },
    });
    try {
      await new Promise<void>((resolve, reject) => {
        exporter.export(spans, (result: ExportResult) => {
          if (result.code === ExportResultCode.SUCCESS) {
            resolve();
          } else {
            reject(result.error ?? new Error(`OTLP span export failed with code ${result.code}`));
          }
        });
      });
    } finally {
      // Drain any pending state so the exporter doesn't keep handles open.
      await exporter.shutdown().catch(() => undefined);
    }
  }

  /**
   * Export OTLP spans to an OSS (non-Databricks) tracking server.
   *
   * `otlpProtoBytes` is an already-serialized OTLP `ExportTraceServiceRequest`
   * protobuf (captured at WAL-write time), POSTed as `application/x-protobuf`
   * with the experiment id forwarded via the `x-mlflow-experiment-id` header.
   */
  async exportOtlpSpans(experimentId: string, otlpProtoBytes: Uint8Array): Promise<void> {
    if (otlpProtoBytes.length === 0) {
      return;
    }
    await makeRawRequest(
      'POST',
      ExportOtlpTracesOss.getEndpoint(this.hostUrl),
      this.headersProvider,
      otlpProtoBytes,
      {
        'Content-Type': 'application/x-protobuf',
        [MLFLOW_EXPERIMENT_ID_HEADER]: experimentId,
      },
    );
  }

  // === TRACE RETRIEVAL METHODS ===
  /**
   * Get a single trace by ID
   * Fetches both trace info and trace data from backend
   * Corresponds to Python: client.get_trace()
   */
  async getTrace(traceId: string): Promise<Trace> {
    const traceInfo = await this.getTraceInfo(traceId);
    const traceData = await this.artifactsClient.downloadTraceData(traceInfo);
    return new Trace(traceInfo, traceData);
  }

  /**
   * Get trace info using V3 API
   * Endpoint: GET /api/3.0/mlflow/traces/{trace_id}
   */
  async getTraceInfo(traceId: string): Promise<TraceInfo> {
    const url = GetTraceInfoV3.getEndpoint(this.hostUrl, traceId);
    const response = await makeRequest<GetTraceInfoV3.Response>('GET', url, this.headersProvider);

    // The V3 API returns a Trace object with trace_info field
    if (response.trace?.trace_info) {
      return TraceInfo.fromJson(response.trace.trace_info);
    }

    throw new Error(`Invalid response format: missing trace_info: ${JSON.stringify(response)}`);
  }

  /**
   * Search for traces that match the given search criteria.
   * Corresponding to the Python SDK's search_traces() method.
   *
   * By default the span data of each matching trace is fetched with one
   * download per trace; traces whose span data cannot be downloaded are
   * skipped, matching the Python SDK's behavior. Pass `includeSpans: false`
   * to return trace metadata only, with empty span data.
   *
   * Locations are passed through to the tracking server, which validates
   * that it supports the requested location types.
   *
   * @param options.locations Trace locations (e.g. MLflow experiments) to search over.
   * @param options.filter Search filter string, e.g. `"tags.env = 'prod'"`.
   * @param options.maxResults Maximum number of traces to return in a single page.
   * @param options.orderBy List of order-by clauses, e.g. `['timestamp_ms DESC']`.
   * @param options.pageToken Pagination token obtained from a previous searchTraces call.
   * @param options.includeSpans Whether to fetch span data for each trace. Defaults to `true`.
   * @returns The matching traces and, when more results are available, a token for the next page.
   *
   * @example
   * ```typescript
   * const { traces, nextPageToken } = await client.searchTraces({
   *   locations: [
   *     { type: TraceLocationType.MLFLOW_EXPERIMENT, mlflowExperiment: { experimentId: '123' } },
   *   ],
   *   filter: "tags.env = 'prod'",
   *   maxResults: 10,
   *   orderBy: ['timestamp_ms DESC'],
   * });
   * ```
   */
  async searchTraces(options: SearchTracesOptions): Promise<SearchTracesResult> {
    const { locations, includeSpans = true } = options;
    if (!locations?.length) {
      throw new Error('At least one location must be specified for searching traces.');
    }

    const url = SearchTracesV3.getEndpoint(this.hostUrl);
    const payload: SearchTracesV3.Request = {
      locations: locations.map(serializeTraceLocation),
      filter: options.filter,
      max_results: options.maxResults,
      order_by: options.orderBy,
      page_token: options.pageToken,
    };
    const response = await makeRequest<SearchTracesV3.Response>(
      'POST',
      url,
      this.headersProvider,
      payload,
    );
    const traceInfos = (response.traces ?? []).map((trace) => TraceInfo.fromJson(trace));

    let traces: Trace[];
    if (includeSpans) {
      const results = await Promise.all(
        traceInfos.map((traceInfo) => this.downloadTraceDataForSearch(traceInfo)),
      );
      traces = results.filter((trace): trace is Trace => trace !== undefined);
    } else {
      traces = traceInfos.map((traceInfo) => new Trace(traceInfo, new TraceData()));
    }

    return {
      traces,
      nextPageToken: response.next_page_token || undefined,
    };
  }

  /**
   * Fetch span data for a search result. Returns undefined when the data
   * cannot be downloaded (e.g. missing or corrupted) so the trace is dropped
   * from the results with a warning, matching the Python SDK.
   */
  private async downloadTraceDataForSearch(traceInfo: TraceInfo): Promise<Trace | undefined> {
    try {
      const traceData = await this.artifactsClient.downloadTraceData(traceInfo);
      return new Trace(traceInfo, traceData);
    } catch (error) {
      console.warn(`Failed to download trace data for trace ${traceInfo.traceId}:`, error);
      return undefined;
    }
  }

  /**
   * Upload trace data to the artifact store.
   */
  async uploadTraceData(traceInfo: TraceInfo, traceData: TraceData): Promise<void> {
    await this.artifactsClient.uploadTraceData(traceInfo, traceData);
  }

  /**
   * Log feedback on a persisted trace via the V3 assessments API.
   * Corresponds to Python `mlflow.log_feedback`.
   *
   * Databricks Unity Catalog V4 assessment posting is not included in this method.
   */
  async logFeedback(params: {
    traceId: string;
    name?: string;
    value?: FeedbackValueType;
    error?: AssessmentError | string;
    source?: AssessmentSource;
    rationale?: string;
    metadata?: Record<string, string>;
    spanId?: string;
  }): Promise<Feedback> {
    assertFeedbackPayload({ value: params.value, error: params.error });
    const metadata = assertMetadataStringMap(params.metadata);
    const source = params.source
      ? {
          sourceType: standardizeSourceType(params.source.sourceType),
          sourceId: params.source.sourceId || 'default',
        }
      : { sourceType: AssessmentSourceType.CODE, sourceId: 'default' };

    const feedback = new Feedback({
      name: params.name,
      value: params.value,
      error: params.error,
      source,
      rationale: params.rationale,
      metadata,
      spanId: params.spanId,
      traceId: params.traceId,
    });

    const url = CreateAssessment.getEndpoint(this.hostUrl, params.traceId);
    const payload: CreateAssessment.Request = { assessment: feedback.toJson() };
    const response = await makeRequest<CreateAssessment.Response>(
      'POST',
      url,
      this.headersProvider,
      payload,
    );
    return Feedback.fromJson(response.assessment);
  }

  /** Create an issue in an experiment. */
  async createIssue(options: CreateIssueOptions): Promise<Issue> {
    assertIssueId(options.experimentId, 'experimentId');
    assertIssueTimestamp(options.lastSeenTimestamp);
    if (this.isDatabricks) {
      assertDatabricksIssueSeverity(options.severity);
      const lastSeen = options.lastSeenTimestamp ?? (options.sourceRunId ? Date.now() : undefined);
      const payload: DatabricksIssues.CreateRequest = {
        title: options.name,
        description: options.description,
        severity: options.severity.toUpperCase(),
        status: (options.status ?? IssueStatus.PENDING).toUpperCase(),
        source_type: options.sourceRunId ? 'ADHOC_RUN' : 'MANUAL',
        source_id: options.sourceRunId,
        last_seen_time: lastSeen === undefined ? undefined : new Date(lastSeen).toISOString(),
        categories: options.categories,
        metadata: options.metadata,
      };
      const url = new URL(DatabricksIssues.getEndpoint(this.hostUrl, options.experimentId));
      if (options.issueId !== undefined) {
        url.searchParams.set('issue_id', options.issueId);
      }
      const response = await makeRequest<DatabricksIssues.Resource>(
        'POST',
        url.toString(),
        this.headersProvider,
        payload,
      );
      return issueFromDatabricksJson(response, options.experimentId);
    }
    const payload: CreateIssue.Request = {
      experiment_id: options.experimentId,
      name: options.name,
      description: options.description,
      severity: options.severity,
      status: options.status ?? IssueStatus.PENDING,
      categories: options.categories,
      metadata: options.metadata,
      issue_id: options.issueId,
      last_seen_timestamp: options.lastSeenTimestamp,
      source_run_id: options.sourceRunId,
    };
    const response = await makeRequest<CreateIssue.Response>(
      'POST',
      CreateIssue.getEndpoint(this.hostUrl),
      this.headersProvider,
      payload,
    );
    return Issue.fromJson(response.issue);
  }

  /** Get an issue. The experiment ID routes Databricks requests to the parent experiment. */
  async getIssue(issueId: string, experimentId: string): Promise<Issue> {
    assertIssueId(experimentId, 'experimentId');
    assertIssueId(issueId, 'issueId');
    if (this.isDatabricks) {
      const response = await makeRequest<DatabricksIssues.Resource>(
        'GET',
        DatabricksIssues.getEndpoint(this.hostUrl, experimentId, issueId),
        this.headersProvider,
      );
      return issueFromDatabricksJson(response, experimentId);
    }
    const response = await makeRequest<GetIssue.Response>(
      'GET',
      GetIssue.getEndpoint(this.hostUrl, issueId),
      this.headersProvider,
    );
    return Issue.fromJson(response.issue);
  }

  /** Update supplied fields; omitted fields retain their values. */
  async updateIssue(issueId: string, options: UpdateIssueOptions): Promise<Issue> {
    assertIssueId(options.experimentId, 'experimentId');
    assertIssueId(issueId, 'issueId');
    assertIssueTimestamp(options.lastSeenTimestamp);
    const metadataKeys = Object.keys(options.metadata ?? {});
    if (options.deleteMetadataKeys?.some((key) => metadataKeys.includes(key))) {
      throw new Error('Cannot update and delete the same metadata key');
    }
    if (this.isDatabricks) {
      assertDatabricksIssueSeverity(options.severity);
      const fields = {
        title: options.name,
        description: options.description,
        status: options.status?.toUpperCase(),
        severity: options.severity?.toUpperCase(),
        categories: options.categories,
        last_seen_time:
          options.lastSeenTimestamp === undefined
            ? undefined
            : new Date(options.lastSeenTimestamp).toISOString(),
      };
      const paths = Object.entries(fields)
        .filter(([, value]) => value !== undefined)
        .map(([key]) => key);
      for (const key of new Set([...metadataKeys, ...(options.deleteMetadataKeys ?? [])])) {
        if (!key || /[,`]/.test(key)) {
          throw new Error('Metadata update keys cannot be empty or contain commas or backticks');
        }
        paths.push(key.includes('.') ? `metadata.\`${key}\`` : `metadata.${key}`);
      }
      if (!paths.length) {
        return this.getIssue(issueId, options.experimentId);
      }
      const payload: DatabricksIssues.UpdateRequest = {
        ...fields,
        name: `experiments/${options.experimentId}/issues/${issueId}`,
        metadata: metadataKeys.length ? options.metadata : undefined,
      };
      const url = new URL(
        DatabricksIssues.getEndpoint(this.hostUrl, options.experimentId, issueId),
      );
      url.searchParams.set('update_mask', paths.join(','));
      const response = await makeRequest<DatabricksIssues.Resource>(
        'PATCH',
        url.toString(),
        this.headersProvider,
        payload,
      );
      return issueFromDatabricksJson(response, options.experimentId);
    }
    const payload: UpdateIssue.Request = {
      issue_id: issueId,
      name: options.name,
      description: options.description,
      severity: options.severity,
      status: options.status,
      categories: options.categories,
      metadata: options.metadata,
      delete_metadata_keys: options.deleteMetadataKeys,
      last_seen_timestamp: options.lastSeenTimestamp,
    };
    const response = await makeRequest<UpdateIssue.Response>(
      'PATCH',
      UpdateIssue.getEndpoint(this.hostUrl, issueId),
      this.headersProvider,
      payload,
    );
    return Issue.fromJson(response.issue);
  }

  /** Search one page of issues using MLflow filter syntax. */
  async searchIssues(options: SearchIssuesOptions): Promise<SearchIssuesResult> {
    assertIssueId(options.experimentId, 'experimentId');
    if (this.isDatabricks) {
      const payload: DatabricksIssues.SearchRequest = {
        parent: `experiments/${options.experimentId}`,
        statuses: databricksIssueStatuses(options.filter),
        page_size: options.maxResults,
        page_token: options.pageToken,
        view: (options.view ?? 'full').toUpperCase(),
      };
      if (options.orderBy?.length) {
        if (options.orderBy.length !== 1) {
          throw new Error('Databricks issue search supports one orderBy field');
        }
        const match = /^(\w+)(?:\s+(ASC|DESC))?$/i.exec(options.orderBy[0].trim());
        const sortKeys: Record<string, string> = {
          severity: 'SEVERITY',
          last_seen_timestamp: 'LAST_SEEN_TIME',
          last_updated_timestamp: 'UPDATE_TIME',
          created_timestamp: 'CREATE_TIME',
        };
        if (!match || !Object.keys(sortKeys).includes(match[1])) {
          throw new Error(`Unsupported Databricks issue orderBy: ${options.orderBy[0]}`);
        }
        payload.sort_key = sortKeys[match[1]];
        payload.sort_order = match[2]?.toUpperCase() === 'DESC' ? 'DESCENDING' : 'ASCENDING';
      }
      // An empty statuses list means "all" to Databricks, so don't send a contradiction.
      if (!payload.statuses.length) {
        return { issues: [] };
      }
      const response = await makeRequest<DatabricksIssues.SearchResponse>(
        'POST',
        `${DatabricksIssues.getEndpoint(this.hostUrl, options.experimentId)}:search`,
        this.headersProvider,
        payload,
      );
      return {
        issues: (response.issues ?? []).map((issue) =>
          issueFromDatabricksJson(issue, options.experimentId),
        ),
        nextPageToken: response.next_page_token || undefined,
        totalCount: response.total_size === undefined ? undefined : Number(response.total_size),
      };
    }
    const payload: SearchIssues.Request = {
      experiment_id: options.experimentId,
      filter_string: options.filter,
      max_results: options.maxResults,
      order_by: options.orderBy,
      page_token: options.pageToken,
      view: options.view,
    };
    const response = await makeRequest<SearchIssues.Response>(
      'POST',
      SearchIssues.getEndpoint(this.hostUrl),
      this.headersProvider,
      payload,
    );
    return {
      issues: (response.issues ?? []).map((issue) => Issue.fromJson(issue)),
      nextPageToken: response.next_page_token || undefined,
      totalCount: response.total_count === undefined ? undefined : Number(response.total_count),
    };
  }

  // === EXPERIMENT METHODS  ===
  /**
   * Create a new experiment
   */
  async createExperiment(
    name: string,
    artifactLocation?: string,
    tags?: Record<string, string>,
  ): Promise<string> {
    const url = CreateExperiment.getEndpoint(this.hostUrl);
    const payload: CreateExperiment.Request = { name, artifact_location: artifactLocation, tags };
    const response = await makeRequest<CreateExperiment.Response>(
      'POST',
      url,
      this.headersProvider,
      payload,
    );
    return response.experiment_id;
  }

  /**
   * Get an experiment by ID, including its tags. Used to auto-resolve UC
   * trace destinations from a Databricks-linked experiment.
   */
  async getExperiment(
    experimentId: string,
  ): Promise<{ experimentId: string; name: string; tags: Record<string, string> } | null> {
    const url = GetExperiment.getEndpoint(this.hostUrl, experimentId);
    try {
      const response = await makeRequest<GetExperiment.Response>('GET', url, this.headersProvider);
      const exp = response.experiment;
      if (!exp?.experiment_id) {
        return null;
      }
      const tags: Record<string, string> = {};
      for (const tag of exp.tags ?? []) {
        tags[tag.key] = tag.value;
      }
      return { experimentId: exp.experiment_id, name: exp.name, tags };
    } catch (error) {
      if (
        error instanceof MlflowHttpError &&
        (error.status === 404 || error.errorCode === 'RESOURCE_DOES_NOT_EXIST')
      ) {
        return null;
      }
      throw error;
    }
  }

  /**
   * Get an experiment by name.
   */
  async getExperimentByName(name: string): Promise<{ experimentId: string; name: string } | null> {
    const url = GetExperimentByName.getEndpoint(this.hostUrl, name);
    try {
      const response = await makeRequest<GetExperimentByName.Response>(
        'GET',
        url,
        this.headersProvider,
      );
      if (!response.experiment?.experiment_id || !response.experiment?.name) {
        return null;
      }
      return {
        experimentId: response.experiment.experiment_id,
        name: response.experiment.name,
      };
    } catch (error) {
      if (
        error instanceof MlflowHttpError &&
        (error.status === 404 || error.errorCode === 'RESOURCE_DOES_NOT_EXIST')
      ) {
        return null;
      }
      throw error;
    }
  }

  /**
   * Delete an experiment
   */
  async deleteExperiment(experimentId: string): Promise<void> {
    const url = DeleteExperiment.getEndpoint(this.hostUrl);
    const payload: DeleteExperiment.Request = { experiment_id: experimentId };
    await makeRequest<void>('POST', url, this.headersProvider, payload);
  }
}

function assertIssueId(value: string, field: 'experimentId' | 'issueId'): void {
  if (typeof value !== 'string' || !value) {
    throw new Error(`${field} must be a nonempty string`);
  }
}

function assertIssueTimestamp(timestamp?: number): void {
  if (
    timestamp !== undefined &&
    (!Number.isSafeInteger(timestamp) || timestamp < 0 || timestamp > 8.64e15)
  ) {
    throw new Error(
      'lastSeenTimestamp must be nonnegative integer milliseconds within the Date range',
    );
  }
}

function assertDatabricksIssueSeverity(severity?: IssueSeverityName): void {
  if (severity === IssueSeverity.NOT_AN_ISSUE) {
    throw new Error('The Databricks Issues API supports low, medium, and high severity');
  }
}

function issueFromDatabricksJson(resource: DatabricksIssues.Resource, experimentId: string): Issue {
  return new Issue({
    issueId: resource.issue_id,
    experimentId,
    name: resource.title,
    description: resource.description ?? '',
    status: resource.status.toLowerCase() as IssueStatusName,
    severity:
      resource.severity && resource.severity !== 'ISSUE_SEVERITY_UNSPECIFIED'
        ? (resource.severity.toLowerCase() as IssueSeverityName)
        : undefined,
    createdTimestamp: Date.parse(resource.create_time),
    lastUpdatedTimestamp: Date.parse(resource.update_time),
    lastSeenTimestamp:
      resource.last_seen_time === undefined ? undefined : Date.parse(resource.last_seen_time),
    sourceRunId:
      resource.source_type === 'ADHOC_RUN' || resource.source_type === 'SCHEDULED_RUN'
        ? resource.source_id
        : undefined,
    categories: resource.categories ?? [],
    createdBy: resource.created_by,
    lastUpdatedBy: resource.last_updated_by,
    metadata: resource.metadata,
  });
}

function databricksIssueStatuses(filter?: string): string[] {
  const statuses = new Set<string>(Object.values(IssueStatus));
  let remaining = filter?.trim() ?? '';
  while (remaining) {
    // The OSS grammar only has string comparisons joined by AND. Databricks can
    // express the status subset; it has no equivalent source_run_id predicate.
    const comparison =
      /^(?:`([^`]+)`|"([^"]+)"|(\w+))\s*(!=|=)\s*(?:'([^']*)'|"([^"]*)"|(\w+))\s*(?:(AND)\b\s*|$)/i.exec(
        remaining,
      );
    if (!comparison) {
      throw new Error('Issue filters require = or != comparisons joined by AND');
    }
    const field = comparison[1] ?? comparison[2] ?? comparison[3];
    if (field !== 'status') {
      throw new Error(`Databricks issue filters support status only; unsupported field: ${field}`);
    }
    const status = (comparison[5] ?? comparison[6] ?? comparison[7]).trim().toLowerCase();
    if (!(Object.values(IssueStatus) as string[]).includes(status)) {
      throw new Error(`Invalid issue status: ${status}`);
    }
    if (comparison[4] === '=') {
      for (const candidate of statuses) {
        if (candidate !== status) {
          statuses.delete(candidate);
        }
      }
    } else {
      statuses.delete(status);
    }
    remaining = remaining.slice(comparison[0].length).trim();
    if (comparison[8] && !remaining) {
      throw new Error('Issue filters cannot end with AND');
    }
  }
  return [...statuses].map((status) => status.toUpperCase());
}
