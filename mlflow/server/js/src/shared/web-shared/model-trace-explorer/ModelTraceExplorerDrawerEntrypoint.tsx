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
  traceComparison?: ModelTraceExplorerV2DrawerProps['traceComparison'];
  navigationLabel?: ModelTraceExplorerV2DrawerProps['navigationLabel'];
}

export const ModelTraceExplorerDrawer = (props: ModelTraceExplorerDrawerProps): JSX.Element => {
  const context = useModelTraceExplorerContext();

  if (!shouldEnableRedesignedTraceExplorer()) {
    const { traceComparison: _traceComparison, navigationLabel: _navigationLabel, ...legacyProps } = props;
    return <LegacyModelTraceExplorerDrawer {...legacyProps} />;
  }

  return (
    <ModelTraceExplorerV2ContextProvider
      renderExportTracesToDatasetsModal={context.renderExportTracesToDatasetsModal}
      renderAddToReviewQueueDropdown={context.renderAddToReviewQueueDropdown}
      DrawerComponent={context.DrawerComponent}
      drawerWidth={context.drawerWidth}
      getDrawerPortalContainer={context.getDrawerPortalContainer}
      initiallyFullscreen={context.initiallyFullscreen}
    >
      <ModelTraceExplorerV2Drawer {...props} />
    </ModelTraceExplorerV2ContextProvider>
  );
};
