export interface GenAIOverviewActivityPoint {
  timestampMs: number;
  count: number;
}

interface GenAIOverviewChartContextBase {
  label: string;
  startTimeMs: number;
  endTimeMsExclusive: number;
}

export interface GenAIOverviewTraceChartContext extends GenAIOverviewChartContextBase {
  stage: 'trace';
  traceCount: number;
}

export interface GenAIOverviewAnalyzeChartContext extends GenAIOverviewChartContextBase {
  stage: 'analyze';
  issueCounts: {
    high: number;
    medium: number;
    low: number;
  };
}

export interface GenAIOverviewEvalChartContext extends GenAIOverviewChartContextBase {
  stage: 'eval';
  totalScoreCount: number;
  runs: Array<{
    runUuid: string;
    runName: string;
    timestampMs: number;
    scores: Record<string, number | null>;
  }>;
}

export interface GenAIOverviewMonitorChartContext extends GenAIOverviewChartContextBase {
  stage: 'monitor';
  scorerName: string;
  points: Array<{
    timestampMs: number;
    value: number | null;
  }>;
}

export type GenAIOverviewChartContext =
  | GenAIOverviewTraceChartContext
  | GenAIOverviewAnalyzeChartContext
  | GenAIOverviewEvalChartContext
  | GenAIOverviewMonitorChartContext;

export type GenAIOverviewTraceState =
  | { status: 'loading' }
  | { status: 'empty'; activity: GenAIOverviewActivityPoint[] }
  | { status: 'ready'; activity: GenAIOverviewActivityPoint[]; totalCount: number }
  | { status: 'permission-denied' }
  | { status: 'warehouse-required' }
  | { status: 'unavailable' }
  | { status: 'error' };

export type GenAIOverviewStageStatus = GenAIOverviewTraceState | { status: 'not-loaded' };
