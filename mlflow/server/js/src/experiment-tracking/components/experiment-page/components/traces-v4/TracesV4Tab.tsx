import React from 'react';
import { GenericSkeleton } from '@databricks/design-system';
import { shouldEnableModelTraceExplorerCustomTraceView } from '@databricks/web-shared/model-trace-explorer';
import { MonitoringConfigProvider } from '@mlflow/mlflow/src/experiment-tracking/hooks/useMonitoringConfig';
import { TracesV4PageWrapper } from './components/TracesV4PageWrapper';
import { TracesV4PageContent } from './components/TracesV4PageContent';

// Custom View pulls in @a2ui (ESM-only) transitively via ExperimentCustomViewProvider.
// Lazy-load it so consumers that don't need Custom View don't pull @a2ui onto their
// static module graph (mirrors the V3 traces page).
const LazyExperimentCustomViewProvider = React.lazy(() =>
  import('../traces-v3/ExperimentCustomViewProvider').then((module) => ({
    default: module.ExperimentCustomViewProvider,
  })),
);

interface TracesV4TabProps {
  experimentId: string;
  isLoadingExperiment?: boolean;
}

/**
 * Entry point for the V4 traces tab. Thin mount controller: waits for the experiment to resolve,
 * then mounts the page content inside the error/notification wrapper. `resetKey` on the experiment
 * id lets the boundary recover cleanly if the user navigates between experiments.
 *
 * `MonitoringConfigProvider` (mirroring TracesV3View) gives the subtree a stable `dateNow` memoized
 * on `lastRefreshTime`. Without it, `useMonitoringConfig`'s no-provider fallback rebuilds
 * `new Date()` every render, so a relative time range's computed `endTime` changes each render and
 * the controller's clear-on-time-change effect wipes the bulk selection before the user can act.
 */
export const TracesV4Tab = ({ experimentId, isLoadingExperiment }: TracesV4TabProps) => {
  const pageContent = <TracesV4PageContent experimentId={experimentId} />;
  const content = shouldEnableModelTraceExplorerCustomTraceView() ? (
    <React.Suspense fallback={<GenericSkeleton css={{ flex: 1, margin: 16 }} />}>
      <LazyExperimentCustomViewProvider key={experimentId} experimentId={experimentId}>
        {pageContent}
      </LazyExperimentCustomViewProvider>
    </React.Suspense>
  ) : (
    pageContent
  );

  return (
    <TracesV4PageWrapper resetKey={experimentId}>
      <MonitoringConfigProvider>
        {isLoadingExperiment ? <GenericSkeleton css={{ flex: 1, margin: 16 }} /> : content}
      </MonitoringConfigProvider>
    </TracesV4PageWrapper>
  );
};
