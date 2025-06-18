import { render } from '../../common/utils/TestUtils.react18';
import ExperimentPage from './ExperimentPage';

import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import { Provider } from 'react-redux';
import { MemoryRouter } from '../../common/utils/RoutingUtils';

jest.mock('./experiment-page/ExperimentView', () => ({
  ExperimentView: jest.fn(() => <div />),
}));

jest.mock('./experiment-page/ExperimentPage', () => ({
  ExperimentPage: jest.fn(() => <div />),
}));

jest.mock('./ExperimentListView', () => jest.fn(() => <div />));

jest.mock('../../common/utils/RoutingUtils', () => ({
  ...jest.requireActual<typeof import('../../common/utils/RoutingUtils')>('../../common/utils/RoutingUtils'),
  Navigate: jest.fn(() => <div />),
}));

jest.mock('../../common/utils/ActionUtils', () => ({
  getUUID: jest.fn(() => 'action_id'),
}));

describe('HomePage', () => {
  const createMockStore = configureStore([thunk, promiseMiddleware()]);
  const defaultMockState = {
    entities: { experimentsById: {} },
    apis: { action_id: { active: false } },
  };

  const renderPage = () => {
    return render(
      <Provider store={createMockStore(defaultMockState)}>
        <MemoryRouter>
          <ExperimentPage />
        </MemoryRouter>
      </Provider>,
    );
  };

  // test('Fetches experiments on page load', () => {
  //   renderPage();
  //   // eslint-disable-next-line jest/no-standalone-expect
  //   expect(searchExperimentsApi).toBeCalled();
  // });

  // test('If button to delete experiment is pressed then open DeleteExperimentModal', async () => {
  //   renderPage();
  //   await userEvent.click(screen.getAllByTestId('delete-experiment-button')[0]);
  //   expect(screen.getByText(`Delete Experiment ""`)).toBeInTheDocument();
  // });
});
