import { describe, it, expect, jest, beforeAll, beforeEach } from '@jest/globals';
import { fireEvent, screen } from '@testing-library/react';
import { render } from '@databricks/testing-library';
import userEvent from '@testing-library/user-event';
import React from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import {
  ModelTraceExplorerContextProvider,
  type DrawerComponentType,
  type RenderAddToReviewQueueDropdownParams,
  useModelTraceExplorerContext,
} from './ModelTraceExplorerContext';
import { ModelTraceExplorerDrawer } from './ModelTraceExplorerDrawer';
import { ModelTraceExplorerContent } from './ModelTraceExplorerContent';
import type { ModelTraceInfoV3 } from './ModelTrace.types';
import { ModelTraceExplorerRightPaneHeader } from './right-pane/ModelTraceExplorerRightPaneHeader';
import { CustomViewAssistantConnectorProvider } from '../custom-view/assistant/CustomViewAssistantConnector';
import { CustomViewDefinitionProvider } from '../custom-view/CustomViewDefinitionContext';
import { COST_METADATA_KEY, TOKEN_USAGE_METADATA_KEY } from '../constants';
import { MOCK_RETRIEVER_SPAN } from '../ModelTraceExplorer.test-utils';
import { setupTestConfig } from '../../flags/test-utils/setupTestConfig';
import {
  dispatchSingleAssistantEventRpc,
  MFE_GLOBAL_CHAT_ACTION,
} from '../../mfe-services/assistant/dispatchSingleAssistantEventRpc';

const mockCopyText = jest.fn();
jest.mock('use-clipboard-copy', () => ({
  useClipboard: jest.fn(() => ({ copy: mockCopyText })),
}));

jest.mock('../../mfe-services/assistant/dispatchSingleAssistantEventRpc', () => ({
  ...jest.requireActual<typeof import('../../mfe-services/assistant/dispatchSingleAssistantEventRpc')>(
    '../../mfe-services/assistant/dispatchSingleAssistantEventRpc',
  ),
  dispatchSingleAssistantEventRpc: jest.fn(),
}));

// The view bodies are heavy and unrelated to this drawer-to-content composition test.
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

const Wrapper = ({ children }: { children: React.ReactNode }) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>{children}</DesignSystemProvider>
  </IntlProvider>
);

// Assert drawer placement via an injected test Drawer that surfaces the
// `position` prop as `data-position`, instead of scraping injected Emotion CSS
// (brittle, and `document.querySelectorAll` is banned in tests). The real
// design-system Drawer's left/right rendering is covered in the design-system
// package; here we only verify the props this component drives from its state.
const TestDrawerComponent: DrawerComponentType = {
  Root: ({ children }) => <div>{children}</div>,
  Content: ({ position, title, width, children }) => (
    <div role="dialog" data-position={position ?? 'right'} data-width={width}>
      <div>{title}</div>
      {children}
    </div>
  ),
};

const HeaderActionsConsumer = () => {
  const { rightPaneHeaderActions } = useModelTraceExplorerContext();
  return <div>{rightPaneHeaderActions}</div>;
};

const ReviewQueueDropdownStub = ({ children }: RenderAddToReviewQueueDropdownParams) => <>{children}</>;

const firePointerEvent = (element: Element, type: string, clientX: number) => {
  const event = new MouseEvent(type, { bubbles: true, cancelable: true, clientX });
  Object.defineProperty(event, 'pointerId', { value: 1 });
  fireEvent(element, event);
};

const mockTraceInfo = {
  trace_id: 'tr-test',
  trace_location: {
    type: 'MLFLOW_EXPERIMENT',
    mlflow_experiment: { experiment_id: 'experiment-id' },
  },
  request_time: '2025-01-01T00:00:00Z',
  execution_duration: '2.201498101s',
  state: 'OK',
  trace_metadata: {
    [TOKEN_USAGE_METADATA_KEY]: JSON.stringify({ input_tokens: 300, output_tokens: 245, total_tokens: 545 }),
    [COST_METADATA_KEY]: JSON.stringify({ input_cost: 0.0001, output_cost: 0.000051, total_cost: 0.000151 }),
  },
  tags: {},
  assessments: [],
} satisfies ModelTraceInfoV3;

const mockUcSchemaTraceInfo = {
  ...mockTraceInfo,
  trace_location: {
    type: 'UC_SCHEMA',
    uc_schema: { catalog_name: 'catalog', schema_name: 'schema' },
  },
  tags: {
    'mlflow.linkedPrompts': JSON.stringify([{ name: 'customer/support?#%', version: '12' }]),
  },
} satisfies ModelTraceInfoV3;

const renderDrawer = ({
  isGeniePanelOpen = false,
  renderAddToReviewQueueDropdown,
  onManagedAddTraceToEvaluationDatasetClick,
  openTraceAssistant,
  isTraceAssistantStreaming,
  openCustomViewAssistant,
  traceInfo = mockTraceInfo,
  renderRightPaneHeader = false,
  selectPreviousEval = jest.fn(),
  selectNextEval = jest.fn(),
}: {
  isGeniePanelOpen?: boolean;
  renderAddToReviewQueueDropdown?: React.ComponentProps<
    typeof ModelTraceExplorerContextProvider
  >['renderAddToReviewQueueDropdown'];
  onManagedAddTraceToEvaluationDatasetClick?: () => void;
  openTraceAssistant?: React.ComponentProps<typeof ModelTraceExplorerContextProvider>['openTraceAssistant'];
  isTraceAssistantStreaming?: boolean;
  openCustomViewAssistant?: () => void;
  traceInfo?: ModelTraceInfoV3;
  renderRightPaneHeader?: boolean;
  selectPreviousEval?: () => void;
  selectNextEval?: () => void;
} = {}) =>
  render(
    <CustomViewAssistantConnectorProvider connector={{ openAssistant: openCustomViewAssistant }}>
      <CustomViewDefinitionProvider views={[]} isLoaded onPersistView={() => Promise.resolve()} canModifyPersistedViews>
        <ModelTraceExplorerContextProvider
          isGeniePanelOpen={isGeniePanelOpen}
          DrawerComponent={TestDrawerComponent}
          renderAddToReviewQueueDropdown={renderAddToReviewQueueDropdown}
          openTraceAssistant={openTraceAssistant}
          isTraceAssistantStreaming={isTraceAssistantStreaming}
        >
          <ModelTraceExplorerDrawer
            selectPreviousEval={selectPreviousEval}
            selectNextEval={selectNextEval}
            isPreviousAvailable
            isNextAvailable
            handleClose={jest.fn()}
            experimentId="experiment-id"
            traceInfo={traceInfo}
            onManagedAddTraceToEvaluationDatasetClick={onManagedAddTraceToEvaluationDatasetClick}
          >
            <HeaderActionsConsumer />
            {renderRightPaneHeader && (
              <ModelTraceExplorerRightPaneHeader
                activeSpan={MOCK_RETRIEVER_SPAN}
                modelTraceInfo={traceInfo}
                showAssessmentsToggle={false}
              />
            )}
            <ModelTraceExplorerContent modelTraceInfo={traceInfo} />
          </ModelTraceExplorerDrawer>
        </ModelTraceExplorerContextProvider>
      </CustomViewDefinitionProvider>
    </CustomViewAssistantConnectorProvider>,
    { wrapper: Wrapper },
  );

describe('ModelTraceExplorerDrawer', () => {
  const { setSafex } = setupTestConfig();

  beforeAll(() => {
    // jsdom does not implement pointer capture; stub so the resize handlers don't throw.
    HTMLElement.prototype.setPointerCapture = jest.fn();
    HTMLElement.prototype.hasPointerCapture = jest.fn(() => true);
    HTMLElement.prototype.releasePointerCapture = jest.fn();
  });

  beforeEach(() => {
    mockCopyText.mockClear();
    jest.mocked(dispatchSingleAssistantEventRpc).mockClear();
    Object.defineProperty(window, 'innerWidth', { configurable: true, value: 1000 });
  });

  it('renders trace metadata and actions in one drawer header row', async () => {
    const user = userEvent.setup();
    Object.defineProperty(window, 'innerWidth', { configurable: true, value: 1800 });
    renderDrawer({
      onManagedAddTraceToEvaluationDatasetClick: jest.fn(),
      renderAddToReviewQueueDropdown: ReviewQueueDropdownStub,
    });

    expect(screen.getByText('Trace')).toBeInTheDocument();
    expect(screen.getByText('test')).toBeInTheDocument();
    expect(screen.getByText('Success')).toBeInTheDocument();
    expect(screen.getByText('2201.50ms')).toBeInTheDocument();
    expect(screen.getByText('545 tokens')).toBeInTheDocument();
    expect(screen.getByText('$0.00')).toBeInTheDocument();
    await user.hover(screen.getByText('545 tokens'));
    expect(await screen.findByText('Usage breakdown')).toBeInTheDocument();
    await user.unhover(screen.getByText('545 tokens'));
    await user.hover(screen.getByText('$0.00'));
    expect(await screen.findByText('Cost breakdown')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Previous trace' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Next trace' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Full screen' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Find in trace' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Add to dataset' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Flag for review' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Share' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Default view' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Analyze trace' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Close trace panel' })).toBeInTheDocument();
  });

  it('does not render linked prompts as a standalone drawer header control', () => {
    renderDrawer({
      traceInfo: {
        ...mockTraceInfo,
        tags: {
          'mlflow.linkedPrompts': JSON.stringify([
            { name: 'customer-support', version: '12' },
            { name: 'request-router', version: '4' },
          ]),
        },
      },
    });

    expect(screen.queryByRole('button', { name: '2 linked prompts' })).not.toBeInTheDocument();
  });

  // ui-test-level: integration. The drawer, context, right-pane header, and linked-prompts popover
  // must be composed for the experiment ID to reach the rendered prompt link.
  it('uses the drawer experiment ID for linked prompts on UC schema traces', async () => {
    renderDrawer({ traceInfo: mockUcSchemaTraceInfo, renderRightPaneHeader: true });

    await userEvent.click(screen.getByRole('button', { name: '1 linked prompt in trace metadata' }));

    expect(await screen.findByRole('link', { name: 'customer/support?#%' })).toHaveAttribute(
      'href',
      expect.stringContaining('/experiments/experiment-id/prompts/customer%2Fsupport%3F%23%25?promptVersion=12'),
    );
  });

  it('progressively simplifies header content as the drawer narrows', () => {
    Object.defineProperty(window, 'innerWidth', { configurable: true, value: 1000 });
    const { unmount } = renderDrawer();

    expect(screen.queryByText('Success')).not.toBeInTheDocument();
    expect(screen.queryByText('2201.50ms')).not.toBeInTheDocument();
    expect(screen.getByText('Default view')).toBeInTheDocument();
    expect(screen.getByText('Analyze trace')).toBeInTheDocument();

    unmount();
    Object.defineProperty(window, 'innerWidth', { configurable: true, value: 800 });
    renderDrawer();

    expect(screen.queryByText('Default view')).not.toBeInTheDocument();
    expect(screen.queryByText('Analyze trace')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Default view' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Analyze trace' })).toBeInTheDocument();
  });

  it('shows trace metadata and action labels in full screen', async () => {
    renderDrawer({
      onManagedAddTraceToEvaluationDatasetClick: jest.fn(),
      renderAddToReviewQueueDropdown: ReviewQueueDropdownStub,
    });

    await userEvent.click(screen.getByRole('button', { name: 'Full screen' }));

    expect(screen.getByText('Success')).toBeInTheDocument();
    expect(screen.getByText('2201.50ms')).toBeInTheDocument();
    expect(screen.getByText('545 tokens')).toBeInTheDocument();
    expect(screen.getByText('$0.00')).toBeInTheDocument();
    expect(screen.getByText('Find')).toBeInTheDocument();
    expect(screen.getByText('Dataset')).toBeInTheDocument();
    expect(screen.getByText('Review')).toBeInTheDocument();
    expect(screen.getByText('Share')).toBeInTheDocument();
  });

  it('copies an absolute link to the trace when clicked', async () => {
    renderDrawer();
    await userEvent.click(await screen.findByRole('button', { name: 'Share' }));
    expect(mockCopyText).toHaveBeenCalledTimes(1);
    // The copied value must be an absolute URL (origin + current path), not a relative path.
    expect(mockCopyText).toHaveBeenCalledWith(expect.stringContaining(window.location.origin));
  });

  it('opens trace analysis without routing it through custom view authoring', async () => {
    const openTraceAssistant = jest.fn();
    renderDrawer({ openTraceAssistant });

    await userEvent.click(screen.getByRole('button', { name: 'Analyze trace' }));

    expect(openTraceAssistant).toHaveBeenCalledTimes(1);
    expect(openTraceAssistant).toHaveBeenCalledWith({
      prompt: 'Analyze trace tr-test.',
      traceInfo: mockTraceInfo,
    });
    expect(dispatchSingleAssistantEventRpc).not.toHaveBeenCalled();
  });

  it('opens trace analysis when the host does not provide a callback', async () => {
    renderDrawer();

    await userEvent.click(screen.getByRole('button', { name: 'Analyze trace' }));

    expect(dispatchSingleAssistantEventRpc).toHaveBeenCalledTimes(1);
    expect(dispatchSingleAssistantEventRpc).toHaveBeenCalledWith({
      type: MFE_GLOBAL_CHAT_ACTION.PAGE_TRIGGERED_CHAT_EVENT,
      payload: {
        message: 'Analyze trace tr-test.',
        messageTags: [],
      },
    });
  });

  it('includes the failed trace ID when opening error analysis', async () => {
    const errorTraceInfo = { ...mockTraceInfo, state: 'ERROR' } satisfies ModelTraceInfoV3;
    renderDrawer({ traceInfo: errorTraceInfo });

    await userEvent.click(screen.getByRole('button', { name: 'Analyze trace' }));

    expect(dispatchSingleAssistantEventRpc).toHaveBeenCalledWith({
      type: MFE_GLOBAL_CHAT_ACTION.PAGE_TRIGGERED_CHAT_EVENT,
      payload: {
        message: 'Debug the error in trace tr-test.',
        messageTags: [],
      },
    });
  });

  it.each(['{ArrowUp}', '{ArrowDown}'])('does not navigate traces when %s is pressed', async (key) => {
    const selectPreviousEval = jest.fn();
    const selectNextEval = jest.fn();
    renderDrawer({ selectPreviousEval, selectNextEval });

    await userEvent.keyboard(key);

    expect(selectPreviousEval).not.toHaveBeenCalled();
    expect(selectNextEval).not.toHaveBeenCalled();
  });

  describe('when the custom trace view flag is off (default fixed drawer)', () => {
    beforeEach(() => {
      setSafex({ [CUSTOM_VIEW_FLAG]: false });
    });

    it('renders a resize handle even when the Genie panel is open', () => {
      renderDrawer({ isGeniePanelOpen: true });
      expect(screen.getByRole('separator', { name: 'Resize trace drawer' })).toBeInTheDocument();
    });

    it('resizes the modal right-snapped drawer from the portaled handle', () => {
      renderDrawer();

      expect(screen.getByRole('dialog')).toHaveAttribute('data-width', '90vw');
      const handle = screen.getByRole('separator', { name: 'Resize trace drawer' });
      firePointerEvent(handle, 'pointerdown', 400);
      firePointerEvent(handle, 'pointermove', 400);
      firePointerEvent(handle, 'pointerup', 400);

      expect(handle).toHaveStyle({ pointerEvents: 'auto' });
      expect(screen.getByRole('dialog')).toHaveAttribute('data-width', '600');
    });

    it('keeps the drawer snapped to the right even when the Genie panel is open', () => {
      renderDrawer({ isGeniePanelOpen: true });
      expect(screen.getByRole('dialog')).toHaveAttribute('data-position', 'right');
    });
  });

  describe('when the custom trace view flag is on but the Genie panel is closed', () => {
    beforeEach(() => {
      setSafex({ [CUSTOM_VIEW_FLAG]: true });
    });

    it('renders a resize handle', () => {
      renderDrawer({ isGeniePanelOpen: false });
      expect(screen.getByRole('separator', { name: 'Resize trace drawer' })).toBeInTheDocument();
    });

    it('keeps the legacy right-snapped placement', () => {
      renderDrawer({ isGeniePanelOpen: false });
      expect(screen.getByRole('dialog')).toHaveAttribute('data-position', 'right');
    });

    it('renders custom view content from the create custom view menu item', async () => {
      const user = userEvent.setup();
      renderDrawer({ openCustomViewAssistant: jest.fn() });

      await user.click(screen.getByRole('button', { name: 'Default view' }));
      await user.click(await screen.findByRole('menuitem', { name: 'Create custom view' }));

      expect(await screen.findByText('custom-view')).toBeInTheDocument();
    });
  });

  describe('when the custom trace view flag is on and the Genie panel is open', () => {
    beforeEach(() => {
      setSafex({ [CUSTOM_VIEW_FLAG]: true });
    });

    it('renders a resize handle', () => {
      renderDrawer({ isGeniePanelOpen: true });
      expect(screen.getByRole('separator', { name: 'Resize trace drawer' })).toBeInTheDocument();
    });

    it('resizes the left-snapped drawer from the portaled handle', () => {
      renderDrawer({ isGeniePanelOpen: true });

      const handle = screen.getByRole('separator', { name: 'Resize trace drawer' });
      firePointerEvent(handle, 'pointerdown', 700);
      firePointerEvent(handle, 'pointermove', 700);
      firePointerEvent(handle, 'pointerup', 700);

      expect(screen.getByRole('dialog')).toHaveAttribute('data-width', '700');
    });

    it('snaps the drawer to the left so the Genie panel stays reachable', () => {
      renderDrawer({ isGeniePanelOpen: true });
      expect(screen.getByRole('dialog')).toHaveAttribute('data-position', 'left');
    });
  });
});
