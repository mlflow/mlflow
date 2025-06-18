import { renderWithIntl } from '../../common/utils/TestUtils.react18';
import ExperimentPage from './ExperimentPage';

import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import { Provider } from 'react-redux';
import { MemoryRouter } from '../../common/utils/RoutingUtils';
import { useFetchExperiments } from './experiment-page/hooks/useFetchExperiments';

jest.mock('../actions', () => ({
  searchDatasetsApi: jest.fn(() => ({ type: 'searchDatasetsApi', payload: Promise.resolve(null) })),
}));

jest.mock('./experiment-page/hooks/useFetchExperiments', () => ({
  useFetchExperiments: jest.fn(() => ({
    fetchExperiments: jest.fn(),
    isLoadingExperiment: false,
  })),
}));

describe('HomePage', () => {
  const createMockStore = configureStore([thunk, promiseMiddleware()]);
  const defaultMockState = {
    entities: { experimentsById: {} },
    apis: { action_id: { active: false } },
  };

  const renderPage = () => {
    return renderWithIntl(
      <Provider store={createMockStore(defaultMockState)}>
        <MemoryRouter>
          <ExperimentPage />
        </MemoryRouter>
      </Provider>,
    );
  };

  test('Fetches experiment on page load', () => {
    renderPage();
    expect(useFetchExperiments).toBeCalled();
  });
});
