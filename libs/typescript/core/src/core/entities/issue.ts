export const IssueStatus = {
  PENDING: 'pending',
  REJECTED: 'rejected',
  RESOLVED: 'resolved',
} as const;

export type IssueStatusName = (typeof IssueStatus)[keyof typeof IssueStatus];

export const IssueSeverity = {
  NOT_AN_ISSUE: 'not_an_issue',
  LOW: 'low',
  MEDIUM: 'medium',
  HIGH: 'high',
} as const;

export type IssueSeverityName = (typeof IssueSeverity)[keyof typeof IssueSeverity];

export const IssueCategory = {
  CORRECTNESS: 'correctness',
  LATENCY: 'latency',
  EXECUTION: 'execution',
  ADHERENCE: 'adherence',
  RELEVANCE: 'relevance',
  SAFETY: 'safety',
  COST: 'cost',
} as const;

export type IssueCategoryName = (typeof IssueCategory)[keyof typeof IssueCategory];

export interface SerializedIssue {
  issue_id: string;
  experiment_id: string;
  name: string;
  description?: string;
  status: IssueStatusName;
  created_timestamp: number | string;
  last_updated_timestamp: number | string;
  severity?: IssueSeverityName;
  root_causes?: string[];
  source_run_id?: string;
  categories?: string[];
  created_by?: string;
  last_updated_by?: string;
  trace_count?: number;
  last_seen_timestamp?: number | string;
  metadata?: Record<string, string>;
}

export class Issue {
  issueId: string;
  experimentId: string;
  name: string;
  description: string;
  status: IssueStatusName;
  createdTimestamp: number;
  lastUpdatedTimestamp: number;
  severity?: IssueSeverityName;
  /** @deprecated Use description and trace evidence for causal explanation. */
  rootCauses?: string[];
  sourceRunId?: string;
  categories?: string[];
  createdBy?: string;
  lastUpdatedBy?: string;
  traceCount?: number;
  lastSeenTimestamp?: number;
  metadata?: Record<string, string>;

  constructor(params: {
    issueId: string;
    experimentId: string;
    name: string;
    description: string;
    status: IssueStatusName;
    createdTimestamp: number;
    lastUpdatedTimestamp: number;
    severity?: IssueSeverityName;
    rootCauses?: string[];
    sourceRunId?: string;
    categories?: string[];
    createdBy?: string;
    lastUpdatedBy?: string;
    traceCount?: number;
    lastSeenTimestamp?: number;
    metadata?: Record<string, string>;
  }) {
    this.issueId = params.issueId;
    this.experimentId = params.experimentId;
    this.name = params.name;
    this.description = params.description;
    this.status = params.status;
    this.createdTimestamp = params.createdTimestamp;
    this.lastUpdatedTimestamp = params.lastUpdatedTimestamp;
    this.severity = params.severity;
    this.rootCauses = params.rootCauses;
    this.sourceRunId = params.sourceRunId;
    this.categories = params.categories;
    this.createdBy = params.createdBy;
    this.lastUpdatedBy = params.lastUpdatedBy;
    this.traceCount = params.traceCount;
    this.lastSeenTimestamp = params.lastSeenTimestamp;
    this.metadata = params.metadata;
  }

  toJson(): SerializedIssue {
    return {
      issue_id: this.issueId,
      experiment_id: this.experimentId,
      name: this.name,
      description: this.description,
      status: this.status,
      created_timestamp: this.createdTimestamp,
      last_updated_timestamp: this.lastUpdatedTimestamp,
      severity: this.severity,
      root_causes: this.rootCauses,
      source_run_id: this.sourceRunId,
      categories: this.categories,
      created_by: this.createdBy,
      last_updated_by: this.lastUpdatedBy,
      trace_count: this.traceCount,
      last_seen_timestamp: this.lastSeenTimestamp,
      metadata: this.metadata,
    };
  }

  static fromJson(json: SerializedIssue): Issue {
    return new Issue({
      issueId: json.issue_id,
      experimentId: json.experiment_id,
      name: json.name,
      description: json.description ?? '',
      status: json.status,
      createdTimestamp: Number(json.created_timestamp),
      lastUpdatedTimestamp: Number(json.last_updated_timestamp),
      severity: json.severity,
      rootCauses: json.root_causes,
      sourceRunId: json.source_run_id,
      categories: json.categories,
      createdBy: json.created_by,
      lastUpdatedBy: json.last_updated_by,
      traceCount: json.trace_count,
      lastSeenTimestamp:
        json.last_seen_timestamp != null ? Number(json.last_seen_timestamp) : undefined,
      metadata: json.metadata,
    });
  }
}
