import { afterEach, beforeEach, describe, expect, jest, test } from '@jest/globals';
import { act, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { render } from '@databricks/web-shared/test-utils/render';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { useState } from 'react';
import { createMemoryRouter, RouterProvider } from 'react-router-dom';

import { HashRouter } from '../../genai-traces-table/utils/RoutingUtils';
import { ModelTraceExplorerDrawer } from './ModelTraceExplorerDrawer';
import { CREATE_CUSTOM_VIEW_QUERY_PARAM } from '../constants';
import { CustomViewAssistantConnectorProvider } from '../custom-view/assistant/CustomViewAssistantConnector';
import {
  CustomViewDefinitionProvider,
  useOptionalCustomViewDefinition,
} from '../custom-view/CustomViewDefinitionContext';
import { MAX_CUSTOM_VIEWS_PER_EXPERIMENT, type CustomView } from '../custom-view/customViewDefinition';

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

  describe('create custom view URL intent', () => {
    type DefinitionState = { views: CustomView[]; isLoaded: boolean };
    let setDefinitionState: (state: DefinitionState) => void = () => {};

    const DraftProbe = () => {
      const customViewDefinition = useOptionalCustomViewDefinition();
      return <div>{customViewDefinition?.isDraft ? 'draft-started' : 'no-draft'}</div>;
    };

    const DrawerHost = () => {
      const [definitionState, setState] = useState<DefinitionState>({ views: [], isLoaded: false });
      setDefinitionState = setState;
      return (
        <CustomViewAssistantConnectorProvider connector={{ openAssistant: jest.fn() }}>
          <CustomViewDefinitionProvider
            views={definitionState.views}
            isLoaded={definitionState.isLoaded}
            onPersistView={() => Promise.resolve()}
            canModifyPersistedViews
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

    const renderDrawerAt = (initialEntry: string) => {
      const router = createMemoryRouter([{ path: '*', element: <DrawerHost /> }], { initialEntries: [initialEntry] });
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
