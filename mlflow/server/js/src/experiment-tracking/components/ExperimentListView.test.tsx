import userEvent from '@testing-library/user-event';
import { screen, fireEvent, renderWithIntl } from '../../common/utils/TestUtils.react18';
import { BrowserRouter } from '../../common/utils/RoutingUtils';
import { Provider } from 'react-redux';
import thunk from 'redux-thunk';
import configureStore from 'redux-mock-store';
import promiseMiddleware from 'redux-promise-middleware';
import { ExperimentListView } from './ExperimentListView';
import Fixtures from '../utils/test-utils/Fixtures';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '../../common/utils/reactQueryHooks';
import { useExperimentListQuery } from './experiment-page/hooks/useExperimentListQuery';

jest.mock('./experiment-page/hooks/useExperimentListQuery', () => ({
  useExperimentListQuery: jest.fn(),
  useInvalidateExperimentList: jest.fn(),
}));

const mountComponent = (props: any) => {
  const experiments = props.experiments.slice(0, 25);
  const mockStore = configureStore([thunk, promiseMiddleware()]);
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  jest.mocked(useExperimentListQuery).mockReturnValue({
    data: experiments,
    isLoading: false,
    error: undefined,
    hasNextPage: false,
    hasPreviousPage: false,
    onNextPage: jest.fn(),
    onPreviousPage: jest.fn(),
    refetch: jest.fn(),
  });

  return renderWithIntl(
    <DesignSystemProvider>
      <Provider
        store={mockStore({
          entities: {
            experimentsById: {},
          },
        })}
      >
        <BrowserRouter>
          <QueryClientProvider client={queryClient}>
            <ExperimentListView experiments={experiments} />
          </QueryClientProvider>
        </BrowserRouter>
      </Provider>
    </DesignSystemProvider>,
  );
};

test('If searchInput is set to "Test" then first shown element in experiment list has the title "Test"', () => {
  mountComponent({ experiments: Fixtures.experiments });
  const input = screen.getByTestId('search-experiment-input');
  fireEvent.change(input, {
    target: { value: 'Test' },
  });
  expect(screen.getAllByTestId('experiment-list-item')[0].textContent).toContain('Test');
});

test('If button to create experiment is pressed then open CreateExperimentModal', async () => {
  mountComponent({ experiments: Fixtures.experiments });
  await userEvent.click(screen.getByTestId('create-experiment-button'));
  expect(screen.getByText('Create Experiment')).toBeInTheDocument();
});

test('should render when experiments are empty', () => {
  mountComponent({
    experiments: [],
  });

  // Get the sidebar header as proof that the component rendered
  expect(screen.getByText('No experiments created')).toBeInTheDocument();
});

test('paginated list should not render everything when there are many experiments', () => {
  const keys = Array.from(Array(1000).keys()).map((k) => k.toString());
  const localExperiments = keys.map((k) => Fixtures.createExperiment({ experimentId: k, name: k }));

  mountComponent({
    experiments: localExperiments,
  });
  const selected = screen.getAllByTestId('experiment-list-item');
  expect(selected.length).toBeLessThan(keys.length);
});
