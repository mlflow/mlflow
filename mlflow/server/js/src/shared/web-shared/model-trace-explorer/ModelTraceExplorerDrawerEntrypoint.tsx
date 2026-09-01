import {
  ModelTraceExplorerDrawer as LegacyModelTraceExplorerDrawer,
  type ModelTraceExplorerDrawerProps,
} from './ModelTraceExplorerDrawer';
import { useModelTraceExplorerContext } from './ModelTraceExplorerContext';
import { ModelTraceExplorerDrawer as ModelTraceExplorerV2Drawer } from './v2/ModelTraceExplorerDrawer';
import { ModelTraceExplorerContextProvider as ModelTraceExplorerV2ContextProvider } from './v2/ModelTraceExplorerContext';
import { shouldEnableRedesignedTraceExplorer } from './shouldEnableRedesignedTraceExplorer';

export const ModelTraceExplorerDrawer = (props: ModelTraceExplorerDrawerProps): JSX.Element => {
  const context = useModelTraceExplorerContext();

  if (!shouldEnableRedesignedTraceExplorer()) {
    return <LegacyModelTraceExplorerDrawer {...props} />;
  }

  return (
    <ModelTraceExplorerV2ContextProvider
      renderExportTracesToDatasetsModal={context.renderExportTracesToDatasetsModal}
      renderAddToReviewQueueDropdown={context.renderAddToReviewQueueDropdown}
      DrawerComponent={context.DrawerComponent}
      drawerWidth={context.drawerWidth}
    >
      <ModelTraceExplorerV2Drawer {...props} />
    </ModelTraceExplorerV2ContextProvider>
  );
};

export type { ModelTraceExplorerDrawerProps };
