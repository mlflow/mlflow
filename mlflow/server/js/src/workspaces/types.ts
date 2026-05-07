export type WorkspaceTraceArchivalConfig = {
  location?: string | null;
  retention?: string | null;
};

export type WorkspaceTraceArchivalConfigInput = {
  location?: string;
  retention?: string;
};

export type Workspace = {
  name: string;
  description?: string | null;
  default_artifact_root?: string | null;
  trace_archival_config?: WorkspaceTraceArchivalConfig | null;
};
