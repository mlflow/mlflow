import type { ComponentProps, ReactNode } from 'react';
import type { CSSObject } from '@emotion/react';
import { type ThemeType, useDesignSystemTheme } from '@databricks/design-system';
import { ModelTraceExplorerContextProvider } from '@databricks/web-shared/model-trace-explorer';
import { GenAITracesTableProvider } from '@databricks/web-shared/genai-traces-table';
import { TracesTableView, type TracesTableViewProps } from '@databricks/web-shared/traces-table';
import { TracesV4TraceDrawer, type TracesV4TraceDrawerProps } from './TracesV4TraceDrawer';

export type TracesV4TableSurfaceModelTraceExplorerProviderProps = Omit<
  ComponentProps<typeof ModelTraceExplorerContextProvider>,
  'children'
>;

export type TracesV4TableSurfaceGenAITracesTableProviderProps = Omit<
  ComponentProps<typeof GenAITracesTableProvider>,
  'children'
>;

export interface TracesV4TableSurfaceProviderProps {
  modelTraceExplorer: TracesV4TableSurfaceModelTraceExplorerProviderProps;
  genAITracesTable: TracesV4TableSurfaceGenAITracesTableProviderProps;
}

export interface TracesV4TableSurfaceOverlaySlots {
  deleteModal?: ReactNode;
  actionModals?: ReactNode;
  notificationContainer?: ReactNode;
}

export interface TracesV4TableSurfaceLayoutProps {
  paddingTop: number;
  paddingLeft: number;
  marginBottom: number;
}

export interface TracesV4TableSurfaceProps {
  providerProps: TracesV4TableSurfaceProviderProps;
  tableViewProps: TracesTableViewProps;
  drawerProps: TracesV4TraceDrawerProps;
  layoutProps: TracesV4TableSurfaceLayoutProps;
  overlaySlots?: TracesV4TableSurfaceOverlaySlots;
}

export const getTracesV4TableSurfaceStyles = (
  theme: ThemeType,
  layoutProps: TracesV4TableSurfaceLayoutProps,
): CSSObject => ({
  display: 'flex',
  flexDirection: 'column',
  flex: 1,
  minHeight: 0,
  gap: theme.spacing.md,
  paddingTop: layoutProps.paddingTop,
  paddingLeft: layoutProps.paddingLeft,
  marginBottom: layoutProps.marginBottom,
});

/**
 * The shared V4 traces rendering surface: the two trace-action providers, the table view, the trace
 * drawer and the page's modals. The Traces tab and the evaluation run view compose it with their own
 * controllers.
 */
export const TracesV4TableSurface = ({
  providerProps,
  tableViewProps,
  drawerProps,
  layoutProps,
  overlaySlots,
}: TracesV4TableSurfaceProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <ModelTraceExplorerContextProvider {...providerProps.modelTraceExplorer}>
      <GenAITracesTableProvider {...providerProps.genAITracesTable}>
        {/* Vertical-fill column for the table surface. Horizontal scroll stays inside TracesTable's
            own `<Table scrollable>`, so the toolbar and pagination bar keep to the visible width. */}
        <div css={getTracesV4TableSurfaceStyles(theme, layoutProps)}>
          <TracesTableView {...tableViewProps} />
          {overlaySlots?.deleteModal}
          <TracesV4TraceDrawer {...drawerProps} />
          {overlaySlots?.actionModals}
          {overlaySlots?.notificationContainer}
        </div>
      </GenAITracesTableProvider>
    </ModelTraceExplorerContextProvider>
  );
};
