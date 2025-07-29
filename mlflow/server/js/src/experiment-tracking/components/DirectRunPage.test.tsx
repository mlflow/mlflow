import { Provider } from 'react-redux';
import { useLocation, createMLflowRoutePath } from '../../common/utils/RoutingUtils';
import { testRoute, TestRouter } from '../../common/utils/RoutingTestUtils';
import configureStore from 'redux-mock-store';
import { ErrorWrapper } from '../../common/utils/ErrorWrapper';
import { getRunApi } from '../actions';
import { DirectRunPage } from './DirectRunPage';
import { useEffect } from 'react';
import { renderWithIntl, screen, act } from '../../common/utils/TestUtils.react18';

jest.mock('../../common/components/PageNotFoundView', () => ({
  PageNotFoundView: () => <div>Page not found</div>,
}));

jest.mock('../actions', () => ({
  getRunApi: jest.fn().mockReturnValue({ type: 'getRunApi', payload: Promise.resolve() }),
}));

describe('DirectRunPage', () => {
  let mockLocation: any;
  let mockStore: any;

  const mountComponent = async (runInfosByUuid = {}, runUuid = '') => {
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
      return null;
    };

    renderWithIntl(
      <Provider store={mockStore}>
        <TestRouter
          initialEntries={[createMLflowRoutePath(`/${runUuid}`)]}
          routes={[
            testRoute(
              <>
                <TestComponent />
              </>,
              createMLflowRoutePath('/experiments/:experimentId/runs/:runId'),
            ),
            testRoute(
              <>
                <TestComponent />
                <DirectRunPage />
              </>,
              createMLflowRoutePath('/:runUuid'),
            ),
          ]}
        />
      </Provider>,
    );
  };

  beforeEach(() => {
    mockLocation = {
      pathname: '',
      search: '',
    };
  });

  afterEach(() => {
    jest.restoreAllMocks();
    mockStore?.clearActions();
  });

  test('redirects to main route if run is loaded into store', async () => {
    await mountComponent(
      {
        '321-run-id': { experimentId: '123-exp-id', runUuid: '321-run-id' },
      },
      '321-run-id',
    );

    expect(mockLocation.pathname).toBe(createMLflowRoutePath('/experiments/123-exp-id/runs/321-run-id'));
  });

  test('properly dispatches redux actions for fetching the run', async () => {
    await mountComponent({}, '321-run-id');

    expect(getRunApi).toHaveBeenCalledWith('321-run-id');
    expect(mockStore.getActions()).toEqual(expect.arrayContaining([expect.objectContaining({ type: 'getRunApi' })]));
  });

  test('displays error if run does not exist', async () => {
    // Suppress 404 console error
    jest.spyOn(console, 'error').mockReturnThis();
    // @ts-expect-error TODO(FEINF-4101)
    jest.mocked(getRunApi).mockImplementation(() => ({
      type: 'getRunApi',
      payload: Promise.reject(new ErrorWrapper('', 404)),
    }));

    await act(async () => mountComponent({}, '321-run-id'));

    expect(screen.getByText('Page not found')).toBeInTheDocument();
  });
});
