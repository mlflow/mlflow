import { ExperimentViewHeader, ExperimentViewHeaderSkeleton } from './ExperimentViewHeader';
import { renderWithIntl, act, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import type { ExperimentEntity } from '@mlflow/mlflow/src/experiment-tracking/types';
import userEvent from '@testing-library/user-event';
import { DesignSystemProvider } from '@databricks/design-system';
import { BrowserRouter } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import { Provider } from 'react-redux';
import thunk from 'redux-thunk';
import configureStore from 'redux-mock-store';
import promiseMiddleware from 'redux-promise-middleware';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';

jest.mock('@databricks/design-system', () => {
  const actual = jest.requireActual<typeof import('@databricks/design-system')>('@databricks/design-system');
  const MockBreadcrumb = ({ children }: { children: React.ReactNode }) => <nav>{children}</nav>;
  const MockBreadcrumbItem = ({ children }: { children: React.ReactNode }) => <div>{children}</div>;
  return {
    ...actual,
    Breadcrumb: Object.assign(MockBreadcrumb, { Item: MockBreadcrumbItem }),
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

  const renderComponent = (experiment = defaultExperiment) => {
    const mockStore = configureStore([thunk, promiseMiddleware()]);
    const queryClient = new QueryClient();

    return renderWithIntl(
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <DesignSystemProvider>
            <Provider
              store={mockStore({
                entities: {
                  experimentsById: {},
                },
              })}
            >
              <ExperimentViewHeader experiment={experiment} setEditing={setEditing} />
            </Provider>
          </DesignSystemProvider>
        </BrowserRouter>
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

    it('renders breadcrumb navigation', () => {
      expect(screen.getByText('Experiments')).toBeInTheDocument();
    });

    it('displays share and management buttons', () => {
      expect(screen.getByRole('button', { name: /share/i })).toBeInTheDocument();
      expect(screen.getByTestId('overflow-menu-trigger')).toBeInTheDocument();
    });
  });
});
