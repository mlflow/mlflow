import { describe, it, expect, jest } from '@jest/globals';
import { screen } from '@testing-library/react';
import { render } from '@databricks/testing-library';
import React from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import type { ModelTrace, ModelTraceExplorerView } from './ModelTrace.types';
import { ModelTraceExplorerContent } from './ModelTraceExplorerContent';
import {
  ModelTraceExplorerViewStateContext,
  type ModelTraceExplorerViewState,
} from './ModelTraceExplorerViewStateContext';
import { setupTestConfig } from '../../flags/test-utils/setupTestConfig';
import { ExperimentPermissionsContextProvider } from '../contexts/ExperimentPermissionsContext';
import { CustomViewAssistantConnectorProvider } from '../custom-view/assistant/CustomViewAssistantConnector';
import { ModelTraceExplorerRightPaneHeaderActionsProvider } from './ModelTraceExplorerContext';

// The view bodies are heavy and unrelated to this composition test.
jest.mock('./ModelTraceExplorerDetailView', () => ({
  ModelTraceExplorerDetailView: () => <div>detail-view</div>,
}));
jest.mock('../linked-prompts/ModelTraceExplorerLinkedPromptsView', () => ({
  ModelTraceExplorerLinkedPromptsView: () => <div>linked-prompts-view</div>,
}));
jest.mock('./custom-view/ModelTraceExplorerCustomView', () => ({
  ModelTraceExplorerCustomView: () => <div>custom-view</div>,
}));

const CUSTOM_VIEW_FLAG = 'databricks.fe.mlflow.enableModelTraceExplorerCustomTraceView';

const setActiveView = jest.fn<(view: ModelTraceExplorerView) => void>();

const viewState: ModelTraceExplorerViewState = {
  rootNode: null,
  nodeMap: {},
  activeView: 'detail',
  setActiveView,
  selectedNode: undefined,
  setSelectedNode: () => {},
  activeTab: 'content',
  setActiveTab: () => {},
  showGraph: false,
  setShowGraph: () => {},
  showTimelineTreeGantt: false,
  setShowTimelineTreeGantt: () => {},
  assessmentsPaneExpanded: false,
  setAssessmentsPaneExpanded: () => {},
  isTraceInitialLoading: false,
  assessmentsPaneEnabled: true,
  updatePaneSizeRatios: () => {},
  getPaneSizeRatios: () => ({ detailsSidebar: 0.7, detailsPane: 0.25, graphPane: 0.5 }),
  readOnly: false,
  topLevelNodes: [],
  subscribeToHighlightEvent: () => () => {},
  highlightAssessment: () => {},
};

const renderContent = ({
  withConnector = true,
  canEditExperiment = true,
  /** Which view is selected by the external trace-view controls. */
  activeView = 'detail',
  traceExplorerDisplayMode = 'default',
}: {
  withConnector?: boolean;
  canEditExperiment?: boolean;
  activeView?: ModelTraceExplorerView;
  traceExplorerDisplayMode?: React.ComponentProps<
    typeof ModelTraceExplorerRightPaneHeaderActionsProvider
  >['traceExplorerDisplayMode'];
} = {}) =>
  render(
    <IntlProvider locale="en" messages={{}}>
      <DesignSystemProvider>
        <ExperimentPermissionsContextProvider canEditExperiment={canEditExperiment}>
          <ModelTraceExplorerViewStateContext.Provider value={{ ...viewState, activeView }}>
            <ModelTraceExplorerRightPaneHeaderActionsProvider traceExplorerDisplayMode={traceExplorerDisplayMode}>
              {withConnector ? (
                <CustomViewAssistantConnectorProvider connector={{ openAssistant: () => {} }}>
                  <ModelTraceExplorerContent modelTraceInfo={{} as ModelTrace['info']} />
                </CustomViewAssistantConnectorProvider>
              ) : (
                <ModelTraceExplorerContent modelTraceInfo={{} as ModelTrace['info']} />
              )}
            </ModelTraceExplorerRightPaneHeaderActionsProvider>
          </ModelTraceExplorerViewStateContext.Provider>
        </ExperimentPermissionsContextProvider>
      </DesignSystemProvider>
    </IntlProvider>,
  );

describe('ModelTraceExplorerContent', () => {
  const { setSafex } = setupTestConfig();

  it('renders the detail view without top-level tabs', () => {
    setSafex({ [CUSTOM_VIEW_FLAG]: true });
    renderContent();

    expect(screen.getByText('detail-view')).toBeInTheDocument();
    expect(screen.queryByRole('tab')).not.toBeInTheDocument();
  });

  it('renders linked prompts when selected externally', () => {
    setSafex({ [CUSTOM_VIEW_FLAG]: true });
    renderContent({ activeView: 'prompts' });

    expect(screen.getByText('linked-prompts-view')).toBeInTheDocument();
  });

  it('renders the custom view over linked prompts when custom display mode is selected', async () => {
    setSafex({ [CUSTOM_VIEW_FLAG]: true });
    renderContent({ activeView: 'prompts', traceExplorerDisplayMode: 'custom' });

    expect(await screen.findByText('custom-view')).toBeInTheDocument();
    expect(screen.queryByText('linked-prompts-view')).not.toBeInTheDocument();
  });

  it('renders the custom view when selected externally', async () => {
    setSafex({ [CUSTOM_VIEW_FLAG]: true });
    renderContent({ activeView: 'custom' });

    expect(await screen.findByText('custom-view')).toBeInTheDocument();
  });

  it('renders the detail view when the custom view flag is disabled', () => {
    setSafex({ [CUSTOM_VIEW_FLAG]: false });
    renderContent();

    expect(screen.getByText('detail-view')).toBeInTheDocument();
  });

  it('renders the detail view when no assistant connector is provided', () => {
    setSafex({ [CUSTOM_VIEW_FLAG]: true });
    renderContent({ withConnector: false });

    expect(screen.getByText('detail-view')).toBeInTheDocument();
  });
});
