import { describe, it, expect, jest } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import React from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import type { ModelTrace } from './ModelTrace.types';
import { ModelTraceExplorerContent } from './ModelTraceExplorerContent';
import {
  ModelTraceExplorerViewStateContext,
  type ModelTraceExplorerViewState,
} from './ModelTraceExplorerViewStateContext';
import { CustomViewAssistantConnectorProvider } from './custom-view/assistant/CustomViewAssistantConnector';

// The tab bodies are heavy (the custom view tab pulls in @a2ui) and unrelated
// to the tab-visibility assertions here, so they are stubbed to keep this test
// focused on the "Custom view" tab's gating logic.
jest.mock('./FeatureUtils', () => ({
  ...jest.requireActual<typeof import('./FeatureUtils')>('./FeatureUtils'),
  shouldEnableModelTraceExplorerCustomTraceView: () => true,
}));
jest.mock('./summary-view/ModelTraceExplorerSummaryView', () => ({
  ModelTraceExplorerSummaryView: () => <div>summary-view</div>,
}));
jest.mock('./ModelTraceExplorerDetailView', () => ({
  ModelTraceExplorerDetailView: () => <div>detail-view</div>,
}));
jest.mock('./linked-prompts/ModelTraceExplorerLinkedPromptsView', () => ({
  ModelTraceExplorerLinkedPromptsView: () => <div>linked-prompts-view</div>,
}));
jest.mock('./custom-view/ModelTraceExplorerCustomView', () => ({
  ModelTraceExplorerCustomView: () => <div>custom-view</div>,
}));

const viewState: ModelTraceExplorerViewState = {
  rootNode: null,
  nodeMap: {},
  activeView: 'detail',
  setActiveView: () => {},
  selectedNode: undefined,
  setSelectedNode: () => {},
  setSelectedNodeAndTab: () => {},
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
  getPaneSizeRatios: () => ({ summarySidebar: 0.75, detailsSidebar: 0.7, detailsPane: 0.25, graphPane: 0.5 }),
  readOnly: false,
  topLevelNodes: [],
  subscribeToHighlightEvent: () => () => {},
  highlightAssessment: () => {},
};

const renderContent = ({ withConnector = true }: { withConnector?: boolean } = {}) =>
  render(
    <IntlProvider locale="en" messages={{}}>
      <DesignSystemProvider>
        <ModelTraceExplorerViewStateContext.Provider value={viewState}>
          {withConnector ? (
            <CustomViewAssistantConnectorProvider connector={{ openAssistant: () => {} }}>
              <ModelTraceExplorerContent modelTraceInfo={{} as ModelTrace['info']} />
            </CustomViewAssistantConnectorProvider>
          ) : (
            <ModelTraceExplorerContent modelTraceInfo={{} as ModelTrace['info']} />
          )}
        </ModelTraceExplorerViewStateContext.Provider>
      </DesignSystemProvider>
    </IntlProvider>,
  );

describe('ModelTraceExplorerContent custom view tab', () => {
  it('renders the Custom view tab when a usable assistant connector is provided', () => {
    renderContent();

    expect(screen.getByRole('tab', { name: 'Custom view' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Details & Timeline' })).toBeInTheDocument();
  });

  it('hides the Custom view tab when no assistant connector is provided', () => {
    renderContent({ withConnector: false });

    expect(screen.queryByRole('tab', { name: 'Custom view' })).not.toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Details & Timeline' })).toBeInTheDocument();
  });

  it('hides the Custom view tab when the connector is mounted but has no openAssistant launcher', () => {
    render(
      <IntlProvider locale="en" messages={{}}>
        <DesignSystemProvider>
          <ModelTraceExplorerViewStateContext.Provider value={viewState}>
            <CustomViewAssistantConnectorProvider connector={{}}>
              <ModelTraceExplorerContent modelTraceInfo={{} as ModelTrace['info']} />
            </CustomViewAssistantConnectorProvider>
          </ModelTraceExplorerViewStateContext.Provider>
        </DesignSystemProvider>
      </IntlProvider>,
    );

    expect(screen.queryByRole('tab', { name: 'Custom view' })).not.toBeInTheDocument();
  });
});
