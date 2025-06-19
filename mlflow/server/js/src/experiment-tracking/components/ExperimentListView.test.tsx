import userEvent from '@testing-library/user-event';
import { screen, renderWithIntl } from '../../common/utils/TestUtils.react18';
import { BrowserRouter } from '../../common/utils/RoutingUtils';
import { Provider } from 'react-redux';
import thunk from 'redux-thunk';
import configureStore from 'redux-mock-store';
import promiseMiddleware from 'redux-promise-middleware';
import { ExperimentListView } from './ExperimentListView';
import Fixtures from '../utils/test-utils/Fixtures';
import { DesignSystemProvider } from '@databricks/design-system';

jest.mock('./experiment-page/hooks/useExperimentListQuery', () => ({
  useExperimentListQuery: jest.fn(),
  useInvalidateExperimentList: jest.fn(),
}));

const mountComponent = (props: any) => {
  const experiments = props.experiments.slice(0, 25);
  const mockStore = configureStore([thunk, promiseMiddleware()]);
  const experimentListViewProps = {
    experiments,
    searchFilter: '',
    setSearchFilter: jest.fn(),
    cursorPaginationProps: {
      hasNextPage: false,
      hasPreviousPage: false,
      onNextPage: jest.fn(),
      onPreviousPage: jest.fn(),
      pageSizeSelect: {
        options: [10],
        default: 10,
        onChange: jest.fn(),
      },
    },
  };

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
          <ExperimentListView {...experimentListViewProps} />
        </BrowserRouter>
      </Provider>
    </DesignSystemProvider>,
  );
};

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
