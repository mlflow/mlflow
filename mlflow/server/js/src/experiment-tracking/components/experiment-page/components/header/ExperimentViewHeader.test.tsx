import { jest, describe, beforeEach, it, expect } from '@jest/globals';
import { ExperimentViewHeader, ExperimentViewHeaderSkeleton } from './ExperimentViewHeader';
import { renderWithIntl, act, screen, waitFor } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import type { ExperimentEntity } from '@mlflow/mlflow/src/experiment-tracking/types';
import userEvent from '@testing-library/user-event';
import { DesignSystemProvider } from '@databricks/design-system';
import { BrowserRouter, createMLflowRoutePath, MemoryRouter } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import { Provider } from 'react-redux';
import thunk from 'redux-thunk';
import configureStore from 'redux-mock-store';
import promiseMiddleware from 'redux-promise-middleware';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import { TestRouter, setupTestRouter, testRoute } from '../../../../../common/utils/RoutingTestUtils';

const mockNavigate = jest.fn();

jest.mock('@databricks/design-system', () => {
  const actual = jest.requireActual<typeof import('@databricks/design-system')>('@databricks/design-system');
  const MockBreadcrumb = ({ children }: { children: React.ReactNode }) => <nav>{children}</nav>;
  const MockBreadcrumbItem = ({ children }: { children: React.ReactNode }) => <div>{children}</div>;
  return {
    ...actual,
    Breadcrumb: Object.assign(MockBreadcrumb, { Item: MockBreadcrumbItem }),
  };
});

jest.mock('../../../../../common/utils/FeatureUtils', () => {
  return {
    ...jest.requireActual<typeof import('../../../../../common/utils/FeatureUtils')>(
      '../../../../../common/utils/FeatureUtils',
    ),
    shouldEnableExperimentPageSideTabs: jest.fn().mockReturnValue(true),
    shouldEnableWorkflowBasedNavigation: jest.fn().mockReturnValue(false),
  };
});

jest.mock('../../../../../common/utils/RoutingUtils', () => {
  const actual = jest.requireActual<typeof import('../../../../../common/utils/RoutingUtils')>(
    '../../../../../common/utils/RoutingUtils',
  );
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  };
});

describe('ExperimentViewHeader', () => {
  const defaultExperiment: ExperimentEntity = {
    experimentId: '123',
    name: 'test/experiment/name',
    artifactLocation: 'file:/tmp/mlruns',
    lifecycleStage: 'active',
    allowedActions: [],
    creationTime: 0,
    lastUpdateTime: 0,
    tags: [],
  };

  const setEditing = jest.fn();

  const { history } = setupTestRouter();

  beforeEach(() => {
    mockNavigate.mockClear();
  });

  const renderComponent = (experiment = defaultExperiment, initialPath = '/') => {
    const mockStore = configureStore([thunk, promiseMiddleware()]);
    const queryClient = new QueryClient();
    const Router = initialPath ? MemoryRouter : BrowserRouter;
    const routerProps = initialPath ? { initialEntries: [initialPath] } : {};

    return renderWithIntl(
      <QueryClientProvider client={queryClient}>
        <DesignSystemProvider>
          <Provider
            store={mockStore({
              entities: {
                experimentsById: {},
              },
            })}
          >
            <TestRouter
              routes={[testRoute(<ExperimentViewHeader experiment={experiment} setEditing={setEditing} />, '*')]}
              initialEntries={[initialPath]}
              history={history}
            />
          </Provider>
        </DesignSystemProvider>
      </QueryClientProvider>,
    );
  };

  describe('rendering', () => {
    beforeEach(async () => {
      await act(async () => {
        renderComponent();
      });
    });

    it('displays the last part of the experiment name', () => {
      expect(screen.getByText('name')).toBeInTheDocument();
    });

    it('shows info tooltip with experiment details', async () => {
      await userEvent.click(screen.getByRole('button', { name: 'Info' }));

      const tooltip = await screen.findByTestId('experiment-view-header-info-tooltip-content');
      expect(tooltip).toHaveTextContent('Path: test/experiment/name');
      expect(tooltip).toHaveTextContent('Experiment ID: 123');
      expect(tooltip).toHaveTextContent('Artifact Location: file:/tmp/mlruns');
    });

    it('displays share and management buttons', () => {
      expect(screen.getByRole('button', { name: /share/i })).toBeInTheDocument();
      expect(screen.getByTestId('overflow-menu-trigger')).toBeInTheDocument();
    });
  });

  describe('back button navigation', () => {
    it('navigates to /experiments from experiment tab pages', async () => {
      renderComponent(defaultExperiment, '/experiments/1/traces');

      await waitFor(() => {
        expect(screen.getByTestId('experiment-view-header-back-button')).toBeInTheDocument();
      });

      await userEvent.click(screen.getByTestId('experiment-view-header-back-button'));

      expect(mockNavigate).toHaveBeenCalledWith(createMLflowRoutePath('/experiments'));
    });

    it('navigates to parent path from deeper pages like session details', async () => {
      renderComponent(defaultExperiment, '/experiments/1/chat-sessions/session_1');

      await waitFor(() => {
        expect(screen.getByTestId('experiment-view-header-back-button')).toBeInTheDocument();
      });

      await userEvent.click(screen.getByTestId('experiment-view-header-back-button'));

      expect(mockNavigate).toHaveBeenCalledWith(createMLflowRoutePath('/experiments/1/chat-sessions'));
    });
  });
});
