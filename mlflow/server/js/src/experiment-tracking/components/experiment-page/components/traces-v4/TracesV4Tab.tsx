import { GenericSkeleton } from '@databricks/design-system';
import { MonitoringConfigProvider } from '@mlflow/mlflow/src/experiment-tracking/hooks/useMonitoringConfig';
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
 *
 * `MonitoringConfigProvider` (mirroring TracesV3View) gives the subtree a stable `dateNow` memoized
 * on `lastRefreshTime`. Without it, `useMonitoringConfig`'s no-provider fallback rebuilds
 * `new Date()` every render, so a relative time range's computed `endTime` changes each render and
 * the controller's clear-on-time-change effect wipes the bulk selection before the user can act.
 */
export const TracesV4Tab = ({ experimentId, storageUCSchema, isLoadingExperiment }: TracesV4TabProps) => {
  return (
    <TracesV4PageWrapper resetKey={experimentId}>
      <MonitoringConfigProvider>
        {isLoadingExperiment ? (
          <GenericSkeleton css={{ flex: 1, margin: 16 }} />
        ) : (
          <TracesV4PageContent experimentId={experimentId} storageUCSchema={storageUCSchema} />
        )}
      </MonitoringConfigProvider>
    </TracesV4PageWrapper>
  );
};
