import { afterEach, beforeEach, describe, expect, jest, test } from '@jest/globals';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { render } from '@databricks/web-shared/test-utils/render';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { HashRouter } from '../../genai-traces-table/utils/RoutingUtils';
import { ModelTraceExplorerDrawer } from './ModelTraceExplorerDrawer';

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
});
