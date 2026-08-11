import { GenericSkeleton } from '@databricks/design-system';
import { TracesV4PageWrapper } from './components/TracesV4PageWrapper';
import { TracesV4PageContent } from './components/TracesV4PageContent';

interface TracesV4TabProps {
  experimentId: string;
  /** UC destination path (the `mlflow.experiment.databricksTraceDestinationPath` tag value). */
  storageUCSchema: string;
  isLoadingExperiment?: boolean;
}

/**
 * Entry point for the V4 traces tab. Thin mount controller: waits for the experiment to resolve
 * (the UC schema drives the trace location), then mounts the page content inside the error/
 * notification wrapper. `resetKey` on the experiment id lets the boundary recover cleanly if the
 * user navigates between experiments.
 */
export const TracesV4Tab = ({ experimentId, storageUCSchema, isLoadingExperiment }: TracesV4TabProps) => {
  return (
    <TracesV4PageWrapper resetKey={experimentId}>
      {isLoadingExperiment ? (
        <GenericSkeleton css={{ flex: 1, margin: 16 }} />
      ) : (
        <TracesV4PageContent experimentId={experimentId} storageUCSchema={storageUCSchema} />
      )}
    </TracesV4PageWrapper>
  );
};
