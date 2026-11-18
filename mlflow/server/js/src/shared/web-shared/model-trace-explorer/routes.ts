/**
 * This file contains a subset of mlflow/server/js/src/experiment-tracking/routes.tsx to be used in the model-trace-explorer.
 */
import { createMLflowRoutePath, generatePath } from './RoutingUtils';

// Route path definitions (used in defining route elements)
const createExperimentPageRoutePath = () => createMLflowRoutePath('/experiments/:experimentId');

export const getExperimentPageRoute = (experimentId: string) => {
  return generatePath(createExperimentPageRoutePath(), { experimentId });
};

export const getExperimentPageTracesTabRoute = (experimentId: string) => {
  return `${getExperimentPageRoute(experimentId)}/traces`;
};

// Query params the playground reads to pre-fill itself from a trace ("Open in Playground").
// Defined here (rather than in experiment-tracking) so both the shared trace explorer that
// builds the link and the playground page that consumes it stay in sync.
export const PLAYGROUND_TRACE_ID_QUERY_PARAM = 'traceId';
export const PLAYGROUND_SPAN_ID_QUERY_PARAM = 'spanId';

export const getExperimentPagePlaygroundRoute = (
  experimentId: string,
  { traceId, spanId }: { traceId?: string; spanId?: string } = {},
) => {
  const base = `${getExperimentPageRoute(experimentId)}/playground`;
  const query = new URLSearchParams();
  if (traceId) {
    query.append(PLAYGROUND_TRACE_ID_QUERY_PARAM, traceId);
  }
  if (spanId) {
    query.append(PLAYGROUND_SPAN_ID_QUERY_PARAM, spanId);
  }
  const queryString = query.toString();
  return queryString ? `${base}?${queryString}` : base;
};
