import { describe, expect, it } from '@jest/globals';
import { useEffect } from 'react';
import { DesignSystemProvider } from '@databricks/design-system';
import { Provider } from 'react-redux';
import thunk from 'redux-thunk';
import configureStore from 'redux-mock-store';
import promiseMiddleware from 'redux-promise-middleware';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import { TestRouter, testRoute } from '../../../common/utils/RoutingTestUtils';
import { createMLflowRoutePath } from '../../../common/utils/RoutingUtils';
import { HeaderVisibilityProvider, useHeaderVisibility } from './ExperimentPageHeaderVisibilityContext';
import { ExperimentPageShellExperimentActions } from './ExperimentPageShellExperimentActions';

const experiment = {
  __typename: 'MlflowExperiment',
  experimentId: '123',
  name: 'test/experiment/name',
  artifactLocation: 'file:/tmp/mlruns',
  lifecycleStage: 'active',
  creationTime: '0',
  lastUpdateTime: '0',
  tags: [],
} as any;

const HideHeaderActions = () => {
  const { setHeaderActionsHidden } = useHeaderVisibility();
  useEffect(() => {
    setHeaderActionsHidden(true);
  }, [setHeaderActionsHidden]);
  return null;
};

const renderActions = (path: string, { hideHeaderActions = false } = {}) =>
  renderWithIntl(
    <QueryClientProvider client={new QueryClient()}>
      <DesignSystemProvider>
        <Provider
          store={configureStore([thunk, promiseMiddleware()])({
            entities: { experimentsById: {}, experimentTagsByExperimentId: {} },
          })}
        >
          <HeaderVisibilityProvider>
            {hideHeaderActions && <HideHeaderActions />}
            <TestRouter
              routes={[
                testRoute(
                  <ExperimentPageShellExperimentActions
                    experiment={experiment}
                    pageAction={<button type="button">Page action</button>}
                  />,
                  '*',
                ),
              ]}
              initialEntries={[createMLflowRoutePath(path)]}
            />
          </HeaderVisibilityProvider>
        </Provider>
      </DesignSystemProvider>
    </QueryClientProvider>,
  );

describe('ExperimentPageShellExperimentActions', () => {
  it('shows the info popover, the page action and the Edit / Delete menu', async () => {
    renderActions('/experiments/123/traces');

    expect(await screen.findByRole('button', { name: 'Info' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Page action' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Open header dropdown menu' })).toBeInTheDocument();
  });

  it('leaves out the Edit / Delete menu on prompt details, like the experiment title header', async () => {
    renderActions('/experiments/123/prompts/my-prompt');

    expect(await screen.findByRole('button', { name: 'Info' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Open header dropdown menu' })).not.toBeInTheDocument();
  });

  it('hides the page action and menu when the page hides header actions, but keeps the info popover', async () => {
    renderActions('/experiments/123/review-queue', { hideHeaderActions: true });

    expect(await screen.findByRole('button', { name: 'Info' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Page action' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Open header dropdown menu' })).not.toBeInTheDocument();
  });
});
