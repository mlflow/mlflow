import { Provider } from 'react-redux';
import { Route, useHistory, MemoryRouter } from 'react-router-dom';
import configureStore, { MockStoreEnhanced } from 'redux-mock-store';
import { ErrorWrapper } from '../../common/utils/ErrorWrapper';
import { mountWithIntl } from '../../common/utils/TestUtils';
import { getRunApi } from '../actions';
import { DirectRunPage } from './DirectRunPage';

jest.mock('./PageNotFoundView', () => ({
  PageNotFoundView: () => <div>Page not found</div>,
}));

jest.mock('../actions', () => ({
  getRunApi: jest.fn().mockReturnValue({ type: 'getRunApi', payload: Promise.resolve() }),
}));

describe('DirectRunPage', () => {
  let mockHistory: any;
  // @ts-expect-error TS(2709): Cannot use namespace 'MockStoreEnhanced' as a type... Remove this comment to see the full error message
  let mockStore: MockStoreEnhanced<any>;

  const mountComponent = (runInfosByUuid = {}, runUuid = '') => {
    mockStore = configureStore([])({
      entities: {
        runInfosByUuid,
      },
    });

    const TestComponent = () => {
      mockHistory = useHistory();

      return (
        <Provider store={mockStore}>
          <Route path='/:runUuid' component={DirectRunPage} />
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

    expect(mockHistory.location.pathname).toBe('/experiments/123-exp-id/runs/321-run-id');
  });

  test('properly dispatches redux actions for fetching the run', () => {
    mountComponent({}, '321-run-id');

    expect(getRunApi).toBeCalledWith('321-run-id');
    expect(mockStore.getActions()).toEqual(
      expect.arrayContaining([expect.objectContaining({ type: 'getRunApi' })]),
    );
  });

  test('displays error if run does not exist', async () => {
    (getRunApi as jest.Mock).mockReturnValue({
      type: 'getRunApi',
      payload: Promise.reject(new ErrorWrapper('', 404)),
    });
    const wrapper = mountComponent({}, '321-run-id');
    await new Promise(setImmediate);

    wrapper.update();

    expect(wrapper.html()).toContain('Page not found');
  });
});
