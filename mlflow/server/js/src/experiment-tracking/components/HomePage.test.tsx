import { render, screen } from '../../common/utils/TestUtils.react17';
import HomePage from './HomePage';

import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import { Provider } from 'react-redux';
import { MemoryRouter, Navigate, Route, Routes } from '../../common/utils/RoutingUtils';
import { searchExperimentsApi } from '../actions';
import { DeepPartial } from 'redux';
import { ReduxState } from '../../redux-types';
import { ExperimentView } from './experiment-page/ExperimentView';
import { ExperimentPage } from './experiment-page/ExperimentPage';

jest.mock('./experiment-page/ExperimentView', () => ({
  ExperimentView: jest.fn(() => <div />),
}));

jest.mock('./experiment-page/ExperimentPage', () => ({
  ExperimentPage: jest.fn(() => <div />),
}));

jest.mock('./ExperimentListView', () => jest.fn(() => <div />));

jest.mock('../actions', () => ({
  searchExperimentsApi: jest.fn(() => ({ type: 'searchExperimentsApi' })),
}));

jest.mock('../../common/utils/RoutingUtils', () => ({
  ...jest.requireActual('../../common/utils/RoutingUtils'),
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
  const renderPage = (initialPath = '/', state?: DeepPartial<ReduxState>) => {
    return render(
      <Provider store={createMockStore({ ...defaultMockState, ...state })}>
        <MemoryRouter initialEntries={[initialPath]}>
          <Routes>
            <Route path="/:experimentId" element={<HomePage />} />
            <Route path="*" element={<HomePage />} />
          </Routes>
        </MemoryRouter>
      </Provider>,
    );
  };
  beforeEach(() => {
    jest.mocked(Navigate).mockClear();
    jest.mocked(ExperimentView).mockClear();
  });
  test('Displays proper message when there are no experiments', () => {
    renderPage();
    // eslint-disable-next-line jest/no-standalone-expect
    expect(screen.getByText('No Experiments Exist')).toBeInTheDocument();
  });
  test('Fetches experiments on page load', () => {
    renderPage();
    // eslint-disable-next-line jest/no-standalone-expect
    expect(searchExperimentsApi).toBeCalled();
  });
  test('Navigates to the first active experiment if experiment ID is not provided', () => {
    renderPage('/', {
      entities: {
        experimentsById: {
          100: { lifecycleStage: 'deleted', experimentId: '100' },
          200: { lifecycleStage: 'active', experimentId: '200' },
        },
      },
    });
    // eslint-disable-next-line jest/no-standalone-expect
    expect(Navigate).toHaveBeenCalledWith({ replace: true, to: '/experiments/200' }, {});
  });

  test('Displays experiment view when experiment ID is provided', () => {
    renderPage('/100', {
      entities: {
        experimentsById: {
          100: { lifecycleStage: 'deleted', experimentId: '100' },
          200: { lifecycleStage: 'active', experimentId: '200' },
        },
      },
    });
    // eslint-disable-next-line jest/no-standalone-expect
    expect(Navigate).not.toHaveBeenCalled();
    // eslint-disable-next-line jest/no-standalone-expect
    expect(ExperimentView).toHaveBeenCalled();
  });

  test('Displays experiment view when experiments are compared', () => {
    renderPage('/?experiments=[100,200]', {
      entities: {
        experimentsById: {
          100: { lifecycleStage: 'deleted', experimentId: '100' },
          200: { lifecycleStage: 'active', experimentId: '200' },
        },
      },
    });
    // eslint-disable-next-line jest/no-standalone-expect
    expect(Navigate).not.toHaveBeenCalled();
    // eslint-disable-next-line jest/no-standalone-expect
    expect(ExperimentView).toHaveBeenCalled();
  });

  test('Display loading state and no content while the request is still active', () => {
    renderPage('/', {
      entities: {},
      apis: {},
    });
    // eslint-disable-next-line jest/no-standalone-expect
    expect(screen.getByRole('img')).toBeInTheDocument();
    // eslint-disable-next-line jest/no-standalone-expect
    expect(Navigate).not.toHaveBeenCalled();
    // eslint-disable-next-line jest/no-standalone-expect
    expect(ExperimentView).not.toHaveBeenCalled();
  });
});
