import type { ReactNode } from 'react';

import { CustomViewDefinitionProvider } from '@databricks/web-shared/model-trace-explorer/custom-view';

import { useExperimentCustomViewDefinition } from './hooks/custom-view/useExperimentCustomViewDefinition';

export interface ExperimentCustomViewDefinitionProviderProps {
  experimentId?: string;
  canModifyPersistedViews?: boolean;
  children: ReactNode;
}

/**
 * Loads experiment-scoped custom views without coupling consumers to the
 * assistant integration used for authoring.
 */
export const ExperimentCustomViewDefinitionProvider = ({
  experimentId,
  canModifyPersistedViews = false,
  children,
}: ExperimentCustomViewDefinitionProviderProps): JSX.Element => {
  const { views, isLoaded, persistView, deleteView } = useExperimentCustomViewDefinition(experimentId);

  return (
    <CustomViewDefinitionProvider
      views={views}
      isLoaded={isLoaded}
      onPersistView={canModifyPersistedViews ? persistView : undefined}
      onDeleteView={canModifyPersistedViews ? deleteView : undefined}
      canModifyPersistedViews={canModifyPersistedViews}
    >
      {children}
    </CustomViewDefinitionProvider>
  );
};
