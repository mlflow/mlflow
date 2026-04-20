export type DiagnosticEventPayload = {
  type: string;
  sessionKey?: string;
  costUsd?: number;
  context?: { limit?: number; used?: number };
  model?: string;
  provider?: string;
  durationMs?: number;
  usage?: {
    input?: number;
    output?: number;
    cacheRead?: number;
    cacheWrite?: number;
    total?: number;
  };
};

export function onDiagnosticEvent(_handler: (event: DiagnosticEventPayload) => void): () => void {
  return () => {};
}
