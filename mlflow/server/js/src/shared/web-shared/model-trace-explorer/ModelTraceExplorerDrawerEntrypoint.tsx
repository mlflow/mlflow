import {
  ModelTraceExplorerDrawer as LegacyModelTraceExplorerDrawer,
  type ModelTraceExplorerDrawerProps as LegacyModelTraceExplorerDrawerProps,
} from './ModelTraceExplorerDrawer';
import { useModelTraceExplorerContext } from './ModelTraceExplorerContext';
import {
  ModelTraceExplorerDrawer as ModelTraceExplorerV2Drawer,
  type ModelTraceExplorerDrawerProps as ModelTraceExplorerV2DrawerProps,
} from './v2/ModelTraceExplorerDrawer';
import { ModelTraceExplorerContextProvider as ModelTraceExplorerV2ContextProvider } from './v2/ModelTraceExplorerContext';
import { shouldEnableRedesignedTraceExplorer } from './shouldEnableRedesignedTraceExplorer';

export interface ModelTraceExplorerDrawerProps extends LegacyModelTraceExplorerDrawerProps {
  headerBanner?: ModelTraceExplorerV2DrawerProps['headerBanner'];
  navigationLabel?: ModelTraceExplorerV2DrawerProps['navigationLabel'];
  drawerViewMode?: ModelTraceExplorerV2DrawerProps['drawerViewMode'];
  onDrawerViewModeChange?: ModelTraceExplorerV2DrawerProps['onDrawerViewModeChange'];
  sessionId?: ModelTraceExplorerV2DrawerProps['sessionId'];
  sessionMetrics?: ModelTraceExplorerV2DrawerProps['sessionMetrics'];
  shareUrl?: ModelTraceExplorerV2DrawerProps['shareUrl'];
  sessionNavigationEnabled?: ModelTraceExplorerV2DrawerProps['sessionNavigationEnabled'];
}

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
