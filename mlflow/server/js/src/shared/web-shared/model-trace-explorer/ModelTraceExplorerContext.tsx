import React, { createContext, useContext, useMemo } from 'react';
import type { ReactNode } from 'react';

import { Drawer } from '@databricks/design-system';

import type { ModelTraceInfoV3 } from './ModelTrace.types';

export interface RenderExportTracesToDatasetsModalParams {
  selectedTraceInfos: ModelTraceInfoV3[];
  experimentId: string;
  visible: boolean;
  setVisible: (visible: boolean) => void;
}

export interface RenderAddToReviewQueueDropdownParams {
  selectedTraceInfos: ModelTraceInfoV3[];
  experimentId: string;
  children: React.ReactNode;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  popoverAlign?: 'start' | 'end';
}

export type DrawerComponentType = {
  Root: (props: {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    modal?: boolean;
    children: ReactNode;
  }) => React.ReactElement;
  Content: (props: Drawer.DrawerContentProps & { isFullscreen?: boolean }) => React.ReactElement;
};

export interface AddToDatasetAction {
  openModal: () => void;
}

export interface ModelTraceExplorerContextValue {
  renderExportTracesToDatasetsModal?: (params: RenderExportTracesToDatasetsModalParams) => React.ReactNode;
  renderAddToReviewQueueDropdown?: React.ComponentType<RenderAddToReviewQueueDropdownParams>;
  DrawerComponent: DrawerComponentType;
  /** When set (e.g. by the evaluation review drawer), content can show "Add to dataset" that calls openModal */
  addToDatasetAction?: AddToDatasetAction;
  drawerWidth?: string | number;
  /** Optional boundary used to contain the drawer and its auxiliary UI (new eval runs UI only). */
  getDrawerPortalContainer?: () => HTMLElement;
  /** Opens the trace drawer at the full size of its viewport or scoped container (new eval runs UI only). */
  initiallyFullscreen?: boolean;
  /**
   * The experiment whose traces this host is showing, for reads that are scoped to it rather than
   * to one trace -- currently the issue statuses that decide which issues a trace still shows.
   *
   * Supplied by the host rather than read from the route, because the drawer and the traces table
   * are embedded outside experiment routes too, and a host that knows which experiment it is
   * showing should not have that fact go missing just because the URL does not spell it out.
   */
  experimentId?: string;
  /**
   * The issue the host has selected, where it has one (the Issues tab reads it from the query
   * string). Surfaces that hide dismissed issues keep this one regardless, since arriving from an
   * issue's own page is how its evidence gets reviewed.
   */
  selectedIssueId?: string;
}

const ModelTraceExplorerContext = createContext<ModelTraceExplorerContextValue>({
  renderExportTracesToDatasetsModal: () => null,
  DrawerComponent: Drawer,
  addToDatasetAction: undefined,
});

interface ModelTraceExplorerContextProviderProps {
  children: React.ReactNode;
  renderExportTracesToDatasetsModal?: (params: RenderExportTracesToDatasetsModalParams) => React.ReactNode;
  renderAddToReviewQueueDropdown?: React.ComponentType<RenderAddToReviewQueueDropdownParams>;
  DrawerComponent?: DrawerComponentType;
  drawerWidth?: string | number;
  getDrawerPortalContainer?: () => HTMLElement;
  initiallyFullscreen?: boolean;
  experimentId?: string;
  selectedIssueId?: string;
}

export const ModelTraceExplorerContextProvider: React.FC<ModelTraceExplorerContextProviderProps> = ({
  children,
  renderExportTracesToDatasetsModal,
  renderAddToReviewQueueDropdown,
  DrawerComponent = Drawer,
  drawerWidth,
  getDrawerPortalContainer,
  initiallyFullscreen,
  experimentId,
  selectedIssueId,
}) => {
  const value = useMemo(
    () => ({
      renderExportTracesToDatasetsModal,
      renderAddToReviewQueueDropdown,
      DrawerComponent,
      drawerWidth,
      getDrawerPortalContainer,
      initiallyFullscreen,
      experimentId,
      selectedIssueId,
    }),
    [
      renderExportTracesToDatasetsModal,
      renderAddToReviewQueueDropdown,
      DrawerComponent,
      drawerWidth,
      getDrawerPortalContainer,
      initiallyFullscreen,
      experimentId,
      selectedIssueId,
    ],
  );

  return <ModelTraceExplorerContext.Provider value={value}>{children}</ModelTraceExplorerContext.Provider>;
};

/** Use inside the drawer to expose "Add to dataset" to trace content (e.g. next to Show assessments). */
export const ModelTraceExplorerAddToDatasetProvider: React.FC<{
  openModal: () => void;
  children: ReactNode;
}> = ({ openModal, children }) => {
  const parent = useContext(ModelTraceExplorerContext);
  const value = useMemo(() => ({ ...parent, addToDatasetAction: { openModal } }), [parent, openModal]);
  return <ModelTraceExplorerContext.Provider value={value}>{children}</ModelTraceExplorerContext.Provider>;
};

export const useModelTraceExplorerContext = () => {
  return useContext(ModelTraceExplorerContext);
};
