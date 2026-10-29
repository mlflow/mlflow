import { afterEach, beforeEach, describe, expect, jest, test } from '@jest/globals';
import { act, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { render } from '@databricks/web-shared/test-utils/render';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { useState } from 'react';
import { createMemoryRouter, RouterProvider } from 'react-router-dom';

import { HashRouter } from '../../genai-traces-table/utils/RoutingUtils';
import { COST_METADATA_KEY, CREATE_CUSTOM_VIEW_QUERY_PARAM, TOKEN_USAGE_METADATA_KEY } from '../constants';
import type { ModelTraceInfoV3 } from './ModelTrace.types';
import { type DrawerComponentType, ModelTraceExplorerContextProvider } from './ModelTraceExplorerContext';
import { ModelTraceExplorerDrawer } from './ModelTraceExplorerDrawer';
import { CustomViewAssistantConnectorProvider } from '../custom-view/assistant/CustomViewAssistantConnector';
import {
  CustomViewDefinitionProvider,
  useOptionalCustomViewDefinition,
} from '../custom-view/CustomViewDefinitionContext';
import { MAX_CUSTOM_VIEWS_PER_EXPERIMENT, type CustomView } from '../custom-view/customViewDefinition';
import { shouldEnableNewEvalRunsUi } from '../../../../common/utils/FeatureUtils';

jest.mock('../../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../../common/utils/FeatureUtils')>(
    '../../../../common/utils/FeatureUtils',
  ),
  shouldEnableNewEvalRunsUi: jest.fn(() => false),
}));

// jsdom has no real clipboard, so observe the text handed to `use-clipboard-copy`.
const mockClipboardCopy = jest.fn();
jest.mock('use-clipboard-copy', () => ({
  useClipboard: () => ({ copy: mockClipboardCopy }),
}));

jest.mock('@mlflow/mlflow/src/assistant', () => ({
  __esModule: true,
  useAssistant: () => ({
    canUseAssistant: false,
    isPanelOpen: false,
    openPanel: jest.fn(),
    sendMessageWhenReady: jest.fn(),
  }),
}));

// jsdom only lets the test URL (path prefix included) be set through the History API.
const setBrowserUrl = (url: string) => {
  // eslint-disable-next-line no-restricted-properties
  window.history.replaceState(null, '', url);
};

describe('ModelTraceExplorerDrawer', () => {
  const originalUrl = window.location.href;

  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(() => {
    setBrowserUrl(originalUrl);
  });

  test('copies a link to the trace that keeps the hash route and path prefix', async () => {
    const traceUrl = `${window.location.origin}/mlflow/#/experiments/1/traces?selectedEvaluationId=tr-abc123`;
    setBrowserUrl(traceUrl);

    render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <HashRouter>
            <ModelTraceExplorerDrawer
              selectPreviousEval={jest.fn()}
              selectNextEval={jest.fn()}
              isPreviousAvailable={false}
              isNextAvailable={false}
              handleClose={jest.fn()}
            >
              <div />
            </ModelTraceExplorerDrawer>
          </HashRouter>
        </DesignSystemProvider>
      </IntlProvider>,
    );

    await userEvent.click(screen.getByRole('button', { name: 'Share' }));

    expect(mockClipboardCopy).toHaveBeenCalledWith(traceUrl);
  });

  describe('header metadata', () => {
    const traceInfo = {
      trace_id: 'tr-abc123',
      request_time: '2025-01-01T00:00:00Z',
      execution_duration: '2.201498101s',
      state: 'OK',
      trace_metadata: {
        [TOKEN_USAGE_METADATA_KEY]: JSON.stringify({ input_tokens: 1300, output_tokens: 245, total_tokens: 1545 }),
        [COST_METADATA_KEY]: JSON.stringify({ input_cost: 0.0001, output_cost: 0.000051, total_cost: 0.000151 }),
      },
      tags: {},
    } as unknown as ModelTraceInfoV3;

    const renderDrawerWithTraceInfo = () =>
      render(
        <IntlProvider locale="en">
          <DesignSystemProvider>
            <HashRouter>
              <ModelTraceExplorerDrawer
                selectPreviousEval={jest.fn()}
                selectNextEval={jest.fn()}
                isPreviousAvailable={false}
                isNextAvailable={false}
                handleClose={jest.fn()}
                traceInfo={traceInfo}
              >
                <div />
              </ModelTraceExplorerDrawer>
            </HashRouter>
          </DesignSystemProvider>
        </IntlProvider>,
      );

    test('flag on: formats duration like span durations, abbreviates tokens and rounds the estimated cost to 4 decimals', async () => {
      jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
      renderDrawerWithTraceInfo();
      await userEvent.click(screen.getByRole('button', { name: 'Full screen' }));

      expect(screen.getByText('2.20s')).toBeInTheDocument();
      expect(screen.getByText('1.5K tokens')).toBeInTheDocument();
      expect(screen.getByText('≈$0.0002')).toBeInTheDocument();
    });

    describe('in a narrow window', () => {
      const originalInnerWidth = window.innerWidth;
      beforeEach(() => {
        Object.defineProperty(window, 'innerWidth', { configurable: true, value: 700 });
      });
      afterEach(() => {
        Object.defineProperty(window, 'innerWidth', { configurable: true, value: originalInnerWidth });
      });

      test('flag on: fullscreen keeps compact actions and hides trace metadata', async () => {
        jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
        renderDrawerWithTraceInfo();
        await userEvent.click(screen.getByRole('button', { name: 'Full screen' }));

        expect(screen.queryByText('Find')).not.toBeInTheDocument();
        expect(screen.queryByText('Share')).not.toBeInTheDocument();
        expect(screen.queryByText('2.20s')).not.toBeInTheDocument();
      });

      test('flag off: fullscreen shows action labels and trace metadata', async () => {
        jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(false);
        renderDrawerWithTraceInfo();
        await userEvent.click(screen.getByRole('button', { name: 'Full screen' }));

        expect(screen.getByText('Find')).toBeInTheDocument();
        expect(screen.getByText('Share')).toBeInTheDocument();
        expect(screen.getByText('2201.50ms')).toBeInTheDocument();
      });
    });

    test('flag off: keeps the existing header formats', async () => {
      jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(false);
      renderDrawerWithTraceInfo();
      await userEvent.click(screen.getByRole('button', { name: 'Full screen' }));

      expect(screen.getByText('2201.50ms')).toBeInTheDocument();
      expect(screen.getByText('1,545 tokens')).toBeInTheDocument();
      expect(screen.getByText('$0.00')).toBeInTheDocument();
    });
  });

  describe('dataset and review shortcuts', () => {
    const traceInfo = {
      trace_id: 'tr-abc123',
      request_time: '2025-01-01T00:00:00Z',
      state: 'OK',
      trace_metadata: {},
      tags: {},
    } as unknown as ModelTraceInfoV3;

    const renderDrawerWithActions = () =>
      render(
        <IntlProvider locale="en">
          <DesignSystemProvider>
            <HashRouter>
              <ModelTraceExplorerContextProvider
                renderExportTracesToDatasetsModal={({ visible }) => (visible ? <div>dataset modal</div> : null)}
                renderAddToReviewQueueDropdown={({ children, open }) => (
                  <>
                    {children}
                    {open && (
                      <div role="menu">
                        <button role="menuitem">review menu option</button>
                      </div>
                    )}
                  </>
                )}
              >
                <ModelTraceExplorerDrawer
                  selectPreviousEval={jest.fn()}
                  selectNextEval={jest.fn()}
                  isPreviousAvailable={false}
                  isNextAvailable={false}
                  handleClose={jest.fn()}
                  experimentId="1"
                  traceInfo={traceInfo}
                >
                  <input aria-label="Trace input" />
                </ModelTraceExplorerDrawer>
              </ModelTraceExplorerContextProvider>
            </HashRouter>
          </DesignSystemProvider>
        </IntlProvider>,
      );

    test('flag on: d opens the dataset modal and r opens the review menu, but not while typing', async () => {
      jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
      const user = userEvent.setup();
      renderDrawerWithActions();

      await user.click(screen.getByRole('textbox', { name: 'Trace input' }));
      await user.keyboard('dr');
      expect(screen.queryByText('dataset modal')).not.toBeInTheDocument();
      expect(screen.queryByRole('menuitem', { name: 'review menu option' })).not.toBeInTheDocument();

      screen.getByRole('button', { name: 'Flag for review' }).focus();
      await user.keyboard('r');
      expect(screen.getByRole('menuitem', { name: 'review menu option' })).toBeVisible();

      // A focused option inside the open review menu keeps its keys.
      screen.getByRole('menuitem', { name: 'review menu option' }).focus();
      await user.keyboard('d');
      expect(screen.queryByText('dataset modal')).not.toBeInTheDocument();
    });

    test('flag on: d opens the dataset modal from the drawer', async () => {
      jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
      const user = userEvent.setup();
      renderDrawerWithActions();

      screen.getByRole('button', { name: 'Add to dataset' }).focus();
      await user.keyboard('d');
      expect(screen.getByText('dataset modal')).toBeInTheDocument();
    });

    test('flag on: the review tooltip names its shortcut', async () => {
      jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
      const user = userEvent.setup();
      renderDrawerWithActions();

      await user.hover(screen.getByRole('button', { name: 'Flag for review' }));
      expect(await screen.findByRole('tooltip')).toHaveTextContent('Flag for review · R');
    });

    test('flag off: d and r do nothing', async () => {
      jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(false);
      const user = userEvent.setup();
      renderDrawerWithActions();

      screen.getByRole('button', { name: 'Add to dataset' }).focus();
      await user.keyboard('dr');
      expect(screen.queryByText('dataset modal')).not.toBeInTheDocument();
      expect(screen.queryByRole('menuitem', { name: 'review menu option' })).not.toBeInTheDocument();
    });
  });

  describe('scoped to a container (the MLflow panel beside the open Assistant)', () => {
    const traceInfo = {
      trace_id: 'tr-abc123',
      request_time: '2025-01-01T00:00:00Z',
      execution_duration: '2.201498101s',
      state: 'OK',
      trace_metadata: {},
      tags: {},
    } as unknown as ModelTraceInfoV3;

    // Surfaces the props the drawer drives (modal, position, width, fullscreen) without the design
    // system's Radix portals.
    const TestDrawerComponent: DrawerComponentType = {
      Root: ({ children, modal }) => (
        <div data-testid="test-drawer-root" data-modal={String(modal)}>
          {children}
        </div>
      ),
      Content: ({ position, title, width, children, isFullscreen }) => (
        <div
          role="dialog"
          data-position={position ?? 'right'}
          data-width={width}
          data-fullscreen={String(Boolean(isFullscreen))}
        >
          <div>{title}</div>
          {children}
        </div>
      ),
    };

    let drawerScope: HTMLDivElement;
    beforeEach(() => {
      drawerScope = document.createElement('div');
      drawerScope.getBoundingClientRect = () =>
        ({ left: 100, right: 1100, width: 1000, top: 50, bottom: 650, height: 600, x: 100, y: 50 }) as DOMRect;
      document.body.appendChild(drawerScope);
    });
    afterEach(() => {
      drawerScope.remove();
    });

    const renderScopedDrawer = ({
      drawerWidth,
      initiallyFullscreen,
      handleClose = jest.fn(),
    }: {
      drawerWidth?: number | string;
      initiallyFullscreen?: boolean;
      handleClose?: () => void;
    } = {}) =>
      render(
        <IntlProvider locale="en">
          <DesignSystemProvider>
            <HashRouter>
              <ModelTraceExplorerContextProvider
                DrawerComponent={TestDrawerComponent}
                drawerWidth={drawerWidth}
                getDrawerPortalContainer={() => drawerScope}
                initiallyFullscreen={initiallyFullscreen}
                renderExportTracesToDatasetsModal={() => null}
                renderAddToReviewQueueDropdown={({ children }) => <>{children}</>}
              >
                <ModelTraceExplorerDrawer
                  selectPreviousEval={jest.fn()}
                  selectNextEval={jest.fn()}
                  isPreviousAvailable={false}
                  isNextAvailable={false}
                  handleClose={handleClose}
                  experimentId="1"
                  traceInfo={traceInfo}
                >
                  <div />
                </ModelTraceExplorerDrawer>
              </ModelTraceExplorerContextProvider>
            </HashRouter>
          </DesignSystemProvider>
        </IntlProvider>,
      );

    test('flag on: opens fullscreen within the container with icon-only actions', async () => {
      jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
      renderScopedDrawer({ drawerWidth: '90%', initiallyFullscreen: true });

      expect(screen.getByRole('dialog')).toHaveAttribute('data-fullscreen', 'true');
      expect(screen.getByRole('dialog')).toHaveAttribute('data-width', '100%');
      expect(screen.getByTestId('test-drawer-root')).toHaveAttribute('data-modal', 'false');
      expect(screen.getByRole('button', { name: 'Exit full screen' })).toBeInTheDocument();
      // The 1000px container is wide enough for labels in a viewport-fullscreen drawer, but a scoped
      // drawer keeps its actions icon-only.
      expect(screen.queryByText('Find')).not.toBeInTheDocument();
      expect(screen.queryByText('Share')).not.toBeInTheDocument();
      expect(screen.queryByText('Dataset')).not.toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Share' })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Add to dataset' })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Flag for review' })).toBeInTheDocument();
      // Trace metadata still shows: the container is at least 960px wide.
      expect(screen.getByText('2.20s')).toBeInTheDocument();
      expect(screen.queryByRole('separator', { name: 'Resize trace drawer' })).not.toBeInTheDocument();
    });

    test('flag on: exiting full screen leaves a resizable 90% drawer inside the container', async () => {
      jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
      const user = userEvent.setup();
      renderScopedDrawer({ drawerWidth: '90%', initiallyFullscreen: true });

      await user.click(screen.getByRole('button', { name: 'Exit full screen' }));

      expect(screen.getByRole('dialog')).toHaveAttribute('data-width', '90%');
      expect(screen.getByRole('dialog')).toHaveAttribute('data-fullscreen', 'false');
      expect(screen.getByRole('button', { name: 'Full screen' })).toBeInTheDocument();
      expect(drawerScope).toContainElement(screen.getByRole('separator', { name: 'Resize trace drawer' }));
    });

    test('flag on: waits for the close animation before notifying the host', async () => {
      jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
      jest.useFakeTimers();
      try {
        const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
        const handleClose = jest.fn();
        renderScopedDrawer({ handleClose });

        await user.click(screen.getByRole('button', { name: 'Close trace panel' }));
        expect(screen.getByRole('dialog')).toBeInTheDocument();
        expect(handleClose).not.toHaveBeenCalled();

        act(() => {
          jest.advanceTimersByTime(179);
        });
        expect(handleClose).not.toHaveBeenCalled();
        act(() => {
          jest.advanceTimersByTime(1);
        });
        expect(handleClose).toHaveBeenCalledTimes(1);
      } finally {
        jest.runOnlyPendingTimers();
        jest.useRealTimers();
      }
    });

    test('flag off: ignores the container and opens as a modal, resizable drawer on the page', async () => {
      jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(false);
      renderScopedDrawer({ drawerWidth: 500, initiallyFullscreen: true });

      expect(screen.getByRole('dialog')).toHaveAttribute('data-fullscreen', 'false');
      expect(screen.getByRole('dialog')).toHaveAttribute('data-width', '500');
      expect(screen.getByTestId('test-drawer-root')).toHaveAttribute('data-modal', 'true');
      expect(screen.getByRole('button', { name: 'Full screen' })).toBeInTheDocument();
      const resizeHandle = screen.getByRole('separator', { name: 'Resize trace drawer' });
      expect(drawerScope).not.toContainElement(resizeHandle);
    });
  });

  describe('create custom view URL intent', () => {
    type DefinitionState = { views: CustomView[]; isLoaded: boolean };
    let setDefinitionState: (state: DefinitionState) => void = () => {};

    const DraftProbe = () => {
      const customViewDefinition = useOptionalCustomViewDefinition();
      return (
        <>
          <div>{customViewDefinition?.isDraft ? 'draft-started' : 'no-draft'}</div>
          <div data-testid="active-view-id">{customViewDefinition?.activeViewId ?? 'none'}</div>
        </>
      );
    };

    const DrawerHost = ({
      initialState = { views: [], isLoaded: false },
      autoSelectFirstView,
    }: {
      initialState?: DefinitionState;
      autoSelectFirstView?: boolean;
    }) => {
      const [definitionState, setState] = useState<DefinitionState>(initialState);
      setDefinitionState = setState;
      return (
        <CustomViewAssistantConnectorProvider connector={{ openAssistant: jest.fn() }}>
          <CustomViewDefinitionProvider
            views={definitionState.views}
            isLoaded={definitionState.isLoaded}
            onPersistView={() => Promise.resolve()}
            canModifyPersistedViews
            autoSelectFirstView={autoSelectFirstView}
          >
            <ModelTraceExplorerDrawer
              selectPreviousEval={jest.fn()}
              selectNextEval={jest.fn()}
              isPreviousAvailable={false}
              isNextAvailable={false}
              handleClose={jest.fn()}
            >
              <DraftProbe />
            </ModelTraceExplorerDrawer>
          </CustomViewDefinitionProvider>
        </CustomViewAssistantConnectorProvider>
      );
    };

    const renderDrawerAt = (initialEntry: string, hostProps: React.ComponentProps<typeof DrawerHost> = {}) => {
      const router = createMemoryRouter([{ path: '*', element: <DrawerHost {...hostProps} /> }], {
        initialEntries: [initialEntry],
      });
      const locationChanges = jest.fn();
      router.subscribe(locationChanges);
      render(
        <IntlProvider locale="en">
          <DesignSystemProvider>
            <RouterProvider router={router} />
          </DesignSystemProvider>
        </IntlProvider>,
      );
      return { router, locationChanges };
    };

    test('consumes the intent once after creation becomes eligible', async () => {
      const { router, locationChanges } = renderDrawerAt(
        `/experiments/1/traces?${CREATE_CUSTOM_VIEW_QUERY_PARAM}=true&retained=value#trace`,
      );

      expect(screen.getByText('no-draft')).toBeInTheDocument();
      expect(router.state.location.search).toContain(`${CREATE_CUSTOM_VIEW_QUERY_PARAM}=true`);

      act(() => setDefinitionState({ views: [], isLoaded: true }));

      expect(await screen.findByText('draft-started')).toBeInTheDocument();
      await waitFor(() => expect(router.state.location.search).toBe('?retained=value'));
      expect(router.state.location.pathname).toBe('/experiments/1/traces');
      expect(router.state.location.hash).toBe('#trace');
      expect(router.state.historyAction).toBe('REPLACE');
      expect(locationChanges).toHaveBeenCalledTimes(1);

      act(() => setDefinitionState({ views: [], isLoaded: true }));

      expect(screen.getByText('draft-started')).toBeInTheDocument();
      expect(locationChanges).toHaveBeenCalledTimes(1);
    });

    test('starts the draft instead of auto-selecting a saved view that was already loaded on mount', async () => {
      const savedView: CustomView = {
        id: 'saved-view',
        name: 'Saved view',
        label: 'Saved view',
        instruction: '',
        template: [],
        createdAtMs: 1,
      };
      const { router } = renderDrawerAt(`/experiments/1/traces?${CREATE_CUSTOM_VIEW_QUERY_PARAM}=true`, {
        initialState: { views: [savedView], isLoaded: true },
        autoSelectFirstView: true,
      });

      expect(await screen.findByText('draft-started')).toBeInTheDocument();
      expect(screen.getByTestId('active-view-id')).toHaveTextContent('none');
      await waitFor(() => expect(router.state.location.search).toBe(''));
    });

    test('keeps the intent when loaded definitions reach the view limit', async () => {
      const maxCustomViews: CustomView[] = Array.from(
        { length: MAX_CUSTOM_VIEWS_PER_EXPERIMENT },
        (_unused, index) => ({
          id: `view-${index}`,
          name: `View ${index}`,
          label: `View ${index}`,
          instruction: '',
          template: [],
          createdAtMs: index,
        }),
      );
      const { router, locationChanges } = renderDrawerAt(
        `/experiments/1/traces?${CREATE_CUSTOM_VIEW_QUERY_PARAM}=true`,
      );

      act(() => setDefinitionState({ views: maxCustomViews, isLoaded: true }));

      expect(screen.getByText('no-draft')).toBeInTheDocument();
      expect(router.state.location.search).toContain(`${CREATE_CUSTOM_VIEW_QUERY_PARAM}=true`);
      expect(locationChanges).not.toHaveBeenCalled();
    });
  });
});
