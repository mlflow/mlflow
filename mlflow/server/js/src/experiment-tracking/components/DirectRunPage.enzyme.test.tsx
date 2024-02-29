import { Provider } from 'react-redux';
import { act } from 'react-dom/test-utils';
import { Route, MemoryRouter, useLocation, Routes } from '../../common/utils/RoutingUtils';
import configureStore from 'redux-mock-store';
import { ErrorWrapper } from '../../common/utils/ErrorWrapper';
import { mountWithIntl } from 'common/utils/TestUtils.enzyme';
import { getRunApi } from '../actions';
import { DirectRunPage } from './DirectRunPage';
import { useEffect } from 'react';

jest.mock('../../common/components/PageNotFoundView', () => ({
  PageNotFoundView: () => <div>Page not found</div>,
}));

jest.mock('../actions', () => ({
  getRunApi: jest.fn().mockReturnValue({ type: 'getRunApi', payload: Promise.resolve() }),
}));

describe('DirectRunPage', () => {
  let mockLocation: any;
  let mockStore: any;

  const mountComponent = (runInfosByUuid = {}, runUuid = '') => {
    mockStore = configureStore([])({
      entities: {
        runInfosByUuid,
      },
    });

    const TestComponent = () => {
      const location = useLocation();
      useEffect(() => {
        mockLocation = location;
      }, [location]);

      return (
        <Provider store={mockStore}>
          <Routes>
            <Route path="/experiments/:experimentId/runs/:runId" element={null} />
            <Route path="/:runUuid" element={<DirectRunPage />} />
          </Routes>
        </Provider>
      );
    };

    return mountWithIntl(
      <MemoryRouter initialEntries={[{ pathname: `/${runUuid}` }]}>
        <TestComponent />
      </MemoryRouter>,
    );
  };
  test('redirects to main route if run is loaded into store', () => {
    mountComponent(
      {
        '321-run-id': { experiment_id: '123-exp-id', run_uuid: '321-run-id' },
      },
      '321-run-id',
    );

    expect(mockLocation.pathname).toBe('/experiments/123-exp-id/runs/321-run-id');
  });

  test('properly dispatches redux actions for fetching the run', () => {
    mountComponent({}, '321-run-id');

    expect(getRunApi).toBeCalledWith('321-run-id');
    expect(mockStore.getActions()).toEqual(expect.arrayContaining([expect.objectContaining({ type: 'getRunApi' })]));
  });

  test('displays error if run does not exist', async () => {
    // Suppress 404 console error
    jest.spyOn(console, 'error').mockReturnThis();
    (getRunApi as jest.Mock).mockReturnValue({
      type: 'getRunApi',
      payload: Promise.reject(new ErrorWrapper('', 404)),
    });
    const wrapper = mountComponent({}, '321-run-id');

    await act(async () => {
      await new Promise(setImmediate);
      wrapper.update();
    });

    expect(wrapper.html()).toContain('Page not found');
    jest.restoreAllMocks();
  });
});
