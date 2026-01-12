import invariant from 'invariant';
import { first, isString } from 'lodash';

import type {
  ModelTraceSpan,
  Assessment,
  Expectation,
  Feedback,
  ModelTraceInfoV3,
  ModelTraceLocation,
  ModelTraceLocationMlflowExperiment,
  ModelTraceLocationUcSchema,
  ModelTraceSpanV3,
} from './ModelTrace.types';
import { fetchAPI, getAjaxUrl } from './ModelTraceExplorer.request.utils';
import {
  createTraceV4SerializedLocation,
  parseTraceV4SerializedLocation,
  parseV4TraceId,
} from './ModelTraceExplorer.utils';

export const deleteAssessment = ({ traceId, assessmentId }: { traceId: string; assessmentId: string }) =>
  fetchAPI(getAjaxUrl(`ajax-api/3.0/mlflow/traces/${traceId}/assessments/${assessmentId}`), 'DELETE');

// these fields are set by the server on create
export type CreateAssessmentPayload = {
  assessment: Omit<Assessment, 'assessment_id' | 'create_time' | 'last_update_time'>;
};

export type CreateAssessmentV3Response = {
  assessment: Assessment;
};

export const createAssessment = ({
  payload,
}: {
  payload: CreateAssessmentPayload;
}): Promise<CreateAssessmentV3Response> =>
  fetchAPI(getAjaxUrl(`ajax-api/3.0/mlflow/traces/${payload.assessment.trace_id}/assessments`), 'POST', payload);

export const fetchTraceInfoV3 = ({ traceId }: { traceId: string }) =>
  fetchAPI(getAjaxUrl(`ajax-api/3.0/mlflow/traces/${traceId}`));

export type UpdateAssessmentPayload = {
  // we only support updating these fields
  assessment: {
    feedback?: Feedback;
    expectation?: Expectation;
    rationale?: string;
    metadata?: Record<string, string>;
    assessment_name?: string;
  };
  // comma-separated list of fields to update
  update_mask: string;
};

export type UpdateAssessmentV3Response = {
  assessment: Assessment;
};

export const updateAssessment = ({
  traceId,
  assessmentId,
  payload,
}: {
  traceId: string;
  assessmentId: string;
  payload: UpdateAssessmentPayload;
}) => fetchAPI(getAjaxUrl(`ajax-api/3.0/mlflow/traces/${traceId}/assessments/${assessmentId}`), 'PATCH', payload);

export type CreateAssessmentV4Response = Assessment;

/**
 * Creates an assessment for a trace using the V4 API.
 *
 * @param payload - The assessment data to create
 * @param traceLocation - Location descriptor for the trace. Can be either:
 *   - A ModelTraceLocation object
 *   - A string in one of these formats:
 *     - Experiment location: experiment ID
 *     - UC Schema location: catalog.schema
 * @param sqlWarehouseId - Optional SQL warehouse ID
 */
const createAssessmentV4 = ({
  // prettier-ignore
  payload: {
    assessment,
  },
  traceLocation,
}: {
  payload: CreateAssessmentPayload;
  traceLocation?: ModelTraceLocation | string;
}): Promise<CreateAssessmentV4Response> => {
  invariant(traceLocation, 'Trace location is required for creating assessment via V4 API');
  const serializedLocation = isString(traceLocation) ? traceLocation : createTraceV4SerializedLocation(traceLocation);
  invariant(serializedLocation, 'Trace location could not be resolved');

  const requestBody: Record<string, any> = assessment;

  const queryParams = new URLSearchParams();
  const endpointPath = getAjaxUrl(
    `ajax-api/4.0/mlflow/traces/${serializedLocation}/${assessment.trace_id}/assessments`,
  );
  const urlWithParams = `${endpointPath}?${queryParams.toString()}`;

  return fetchAPI(urlWithParams, 'POST', requestBody);
};

export type UpdateAssessmentV4Response = Assessment;

/**
 * Updates an existing assessment using the V4 API.
 *
 * @param traceId - The ID of the trace containing the assessment
 * @param assessmentId - The ID of the assessment to update
 * @param payload - The assessment fields to update and update mask
 * @param traceLocation - Location descriptor for the trace. Can be either:
 *   - A ModelTraceLocation object
 *   - A string in one of these formats:
 *     - Experiment location: experiment ID
 *     - UC Schema location: catalog.schema
 * @param sqlWarehouseId - Optional SQL warehouse ID
 */
const updateAssessmentV4 = ({
  traceId,
  assessmentId,
  payload,
  traceLocation,
}: {
  traceId: string;
  assessmentId: string;
  payload: UpdateAssessmentPayload;
  traceLocation?: ModelTraceLocation | string;
}) => {
  const { assessment, update_mask } = payload;
  invariant(traceLocation, 'Trace location is required for updating assessment via V4 API');
  const serializedLocation = isString(traceLocation) ? traceLocation : createTraceV4SerializedLocation(traceLocation);
  invariant(serializedLocation, 'Trace location could not be resolved');

  const queryParams = new URLSearchParams();
  queryParams.append('update_mask', update_mask);
  const endpointPath = getAjaxUrl(
    `ajax-api/4.0/mlflow/traces/${serializedLocation}/${traceId}/assessments/${assessmentId}`,
  );
  const urlWithParams = `${endpointPath}?${queryParams.toString()}`;
  return fetchAPI(urlWithParams, 'PATCH', assessment);
};

/**
 * Deletes an assessment using the V4 API.
 *
 * @param traceId - The ID of the trace containing the assessment
 * @param assessmentId - The ID of the assessment to delete
 * @param traceLocation - Location descriptor for the trace. Can be either:
 *   - A ModelTraceLocation object
 *   - A string in one of these formats:
 *     - Experiment location: experiment ID
 *     - UC Schema location: catalog.schema
 * @param sqlWarehouseId - Optional SQL warehouse ID
 */
const deleteAssessmentV4 = ({
  traceId,
  sqlWarehouseId,
  assessmentId,
  traceLocation,
}: {
  traceId: string;
  sqlWarehouseId?: string;
  assessmentId: string;
  traceLocation: ModelTraceLocation | string;
}) => {
  const queryParams = new URLSearchParams();
  invariant(traceLocation, 'Trace location is required for deleting assessment via V4 API');
  const serializedLocation = isString(traceLocation) ? traceLocation : createTraceV4SerializedLocation(traceLocation);
  invariant(serializedLocation, 'Trace location could not be resolved');
  const endpointPath = getAjaxUrl(
    `ajax-api/4.0/mlflow/traces/${serializedLocation}/${traceId}/assessments/${assessmentId}`,
  );
  const urlWithParams = `${endpointPath}?${queryParams.toString()}`;
  return fetchAPI(urlWithParams, 'DELETE');
};

export const searchTracesV4 = async ({
  signal,
  orderBy,
  locations,
  filter,
  pageSize,
}: {
  signal?: AbortSignal;
  orderBy?: string[];
  filter?: string;
  locations?: (ModelTraceLocationMlflowExperiment | ModelTraceLocationUcSchema)[];
  pageSize?: number;
}) => {
  const payload: Record<string, any> = {
    locations,
    filter,
    max_results: pageSize ?? 1000,
    order_by: orderBy,
  };
  const queryResponse = await fetchAPI(getAjaxUrl('ajax-api/4.0/mlflow/traces/search'), 'POST', payload, signal);

  const json = queryResponse as { trace_infos: ModelTraceInfoV3[]; next_page_token?: string };

  return json?.trace_infos ?? [];
};

export const getBatchTracesV4 = async ({
  traceIds,
  traceLocation,
}: {
  /**
   * List of trace IDs to fetch.
   */
  traceIds: string[];
  /**
   * Location descriptor of the traces. All provided trace ID must correspond to and share the same location.
   */
  traceLocation: ModelTraceLocation | string;
}) => {
  const locationString = isString(traceLocation) ? traceLocation : createTraceV4SerializedLocation(traceLocation);

  const endpointPath = getAjaxUrl(`ajax-api/4.0/mlflow/traces/${locationString}/batchGet`);

  const queryParams = new URLSearchParams();
  for (const traceId of traceIds) {
    queryParams.append('trace_ids', traceId);
  }

  const urlWithParams = `${endpointPath}?${queryParams.toString()}`;

  const data: {
    traces: {
      trace_info: ModelTraceInfoV3;
      spans: ModelTraceSpanV3[];
    }[];
  } = await fetchAPI(urlWithParams, 'GET');

  return data;
};

/**
 * Traces API: get a single trace (info + spans) with optional partial support. Only supported
 * by OSS SQLAlchemyStore now.
 */
export const getExperimentTraceV3 = ({ traceId }: { traceId: string }) => {
  const endpointPath = getAjaxUrl(`ajax-api/3.0/mlflow/traces/get`);

  const queryParams = new URLSearchParams();
  queryParams.append('trace_id', traceId);
  queryParams.append('allow_partial', 'true');
  const urlWithParams = `${endpointPath}?${queryParams.toString()}`;

  return fetchAPI(urlWithParams, 'GET');
};

/**
 * Fetch function for V4 traces.
 * TraceV3 is the same format for V4 traces
 */
const getTraceV4 = async (
  traceInput: Partial<ModelTraceInfoV3> | string,
  sqlWarehouseId?: string,
): Promise<{
  info: ModelTraceInfoV3;
  data: {
    spans: ModelTraceSpan[];
  };
}> => {
  let traceId: string;
  let traceLocation: ModelTraceLocation;

  if (typeof traceInput === 'string') {
    const parsed = parseV4TraceId(traceInput);
    invariant(parsed?.trace_id, 'Could not parse trace ID from serialized trace identifier');
    invariant(parsed?.trace_location, 'Could not parse trace location from serialized trace identifier');
    traceId = parsed.trace_id;
    traceLocation = parseTraceV4SerializedLocation(parsed.trace_location);
  } else {
    invariant(traceInput.trace_id, 'Trace ID is required');
    invariant(traceInput.trace_location, 'Trace location is required');

    traceId = traceInput.trace_id;
    traceLocation = traceInput.trace_location;
  }

  const traceResponse = await getBatchTracesV4({
    traceIds: [traceId],
    traceLocation: traceLocation,
  });

  const trace = first(
    // Convert response to the commonly used format
    traceResponse.traces.map((trace) => ({
      info: trace.trace_info,
      data: { spans: trace.spans },
    })),
  );

  invariant(trace, 'No trace found in response');
  return trace;
};

// prettier-ignore
export const getTraceInfoV4 = ({
  traceId,
  traceLocation,
}: {
  traceId: string;
  traceLocation: ModelTraceLocation | string;
}) => {
  const serializedLocation = isString(traceLocation) ? traceLocation : createTraceV4SerializedLocation(traceLocation);
  const endpointPath = getAjaxUrl(`ajax-api/4.0/mlflow/traces/${serializedLocation}/${traceId}/info`);
  const queryParams = new URLSearchParams();

  const urlWithParams = `${endpointPath}?${queryParams.toString()}`;

  return fetchAPI(urlWithParams);
};

export const setTraceTagV4 = async ({
  traceLocation,
  tag,
  traceId,
}: {
  traceLocation: ModelTraceLocation;
  tag: { key: string; value: string };
  traceId: string;
}) => {
  const serializedTraceLocation = createTraceV4SerializedLocation(traceLocation);
  const endpointPath = getAjaxUrl(`ajax-api/4.0/mlflow/traces/${serializedTraceLocation}/${traceId}/tags`);
  const searchParams = new URLSearchParams();
  const queryString = searchParams.toString();
  const uri = `${endpointPath}?${queryString}`;
  return fetchAPI(uri, 'PATCH', tag);
};

export const deleteTraceTagV4 = async ({
  tagKey,
  traceId,
  traceLocation,
}: {
  tagKey: string;
  traceId: string;
  traceLocation: ModelTraceLocation;
}) => {
  const serializedTraceLocation = createTraceV4SerializedLocation(traceLocation);
  const endpointPath = getAjaxUrl(
    `ajax-api/4.0/mlflow/traces/${serializedTraceLocation}/${traceId}/tags/${encodeURIComponent(tagKey)}`,
  );
  const searchParams = new URLSearchParams();
  const queryString = searchParams.toString();
  const uri = `${endpointPath}?${queryString}`;
  return fetchAPI(uri, 'DELETE');
};

export const setTraceTagV3 = async ({ tag, traceId }: { tag: { key: string; value: string }; traceId: string }) => {
  const endpointPath = getAjaxUrl(`ajax-api/3.0/mlflow/traces/${traceId}/tags`);
  return fetchAPI(endpointPath, 'PATCH', tag);
};

export const getTraceV3Data = (traceId: string) => {
  const endpointPath = getAjaxUrl(`ajax-api/3.0/mlflow/get-trace-artifact`);
  const searchParams = new URLSearchParams();
  searchParams.append('request_id', traceId);
  const queryString = searchParams.toString();
  const uri = `${endpointPath}?${queryString}`;
  return fetchAPI(uri, 'GET');
};

export const getTraceV3Info = (traceId: string) => {
  const endpointPath = getAjaxUrl(`ajax-api/3.0/mlflow/traces/${traceId}`);
  return fetchAPI(endpointPath, 'GET');
};

/**
 * Combines two requests (info and data) into a single response to get the full trace.
 */
export const getTraceV3 = async (traceId: string) => {
  const [traceInfoResponse, traceDataResponse] = await Promise.all([getTraceV3Info(traceId), getTraceV3Data(traceId)]);
  const traceInfo = traceInfoResponse?.trace?.trace_info;
  return {
    info: traceInfo,
    data: traceDataResponse,
  };
};

export const deleteTraceTagV3 = async ({ tagKey, traceId }: { tagKey: string; traceId: string }) => {
  const endpointPath = getAjaxUrl(`ajax-api/3.0/mlflow/traces/${traceId}/tags`);
  const searchParams = new URLSearchParams();

  searchParams.append('key', tagKey);

  const queryString = searchParams.toString();
  const uri = `${endpointPath}?${queryString}`;
  return fetchAPI(uri, 'DELETE');
};

/**
 * Service containing methods for interacting with the V4 traces API.
 */
export const TracesServiceV4 = {
  searchTracesV4,
  getBatchTracesV4,
  getTraceV4,
  getTraceInfoV4,
  createAssessmentV4,
  updateAssessmentV4,
  deleteAssessmentV4,
  // END-EDGE
  setTraceTagV4,
  deleteTraceTagV4,
};

/**
 * Service containing methods for interacting with the V3 traces API.
 */
export const TracesServiceV3 = {
  setTraceTagV3,
  deleteTraceTagV3,
  getTraceV3Info,
  getTraceV3Data,
  getTraceV3,
};
