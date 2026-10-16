import { afterEach, beforeEach, describe, expect, jest, test } from '@jest/globals';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { render } from '@databricks/web-shared/test-utils/render';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { HashRouter } from '../../genai-traces-table/utils/RoutingUtils';
import { COST_METADATA_KEY, TOKEN_USAGE_METADATA_KEY } from '../constants';
import type { ModelTraceInfoV3 } from './ModelTrace.types';
import { ModelTraceExplorerContextProvider } from './ModelTraceExplorerContext';
import { ModelTraceExplorerDrawer } from './ModelTraceExplorerDrawer';
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
});
