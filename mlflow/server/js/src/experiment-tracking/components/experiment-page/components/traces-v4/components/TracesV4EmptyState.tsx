import { TracesViewTableNoTracesQuickstart } from '@mlflow/mlflow/src/experiment-tracking/components/traces/quickstart/TracesViewTableNoTracesQuickstart';

/**
 * The V4 tab's no-traces quickstart CTA. Shows the generic non-GenAI quickstart
 * (OSS has no special GenAI experiment support in this context).
 */
export const TracesV4EmptyState = ({ experimentId }: { experimentId: string }) => {
  return <TracesViewTableNoTracesQuickstart baseComponentId="mlflow.traces" />;
};
