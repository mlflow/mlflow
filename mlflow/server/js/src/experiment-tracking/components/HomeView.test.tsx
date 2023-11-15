/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import thunk from 'redux-thunk';
import configureStore from 'redux-mock-store';
import promiseMiddleware from 'redux-promise-middleware';
import { Provider } from 'react-redux';
import { BrowserRouter, MemoryRouter } from '../../common/utils/RoutingUtils';
import Fixtures from '../utils/test-utils/Fixtures';
import HomeView, { getFirstActiveExperiment } from './HomeView';
import { mount } from 'enzyme';

const mockNavigate = jest.fn();

jest.mock('react-router-dom-v5-compat', () => ({
  ...jest.requireActual('react-router-dom-v5-compat'),
  Navigate: (props: any) => {
    mockNavigate(props);
    return null;
  },
}));

const experiments = {
  1: Fixtures.createExperiment({ experiment_id: '1', name: '1', lifecycle_stage: 'deleted' }),
  3: Fixtures.createExperiment({ experiment_id: '3', name: '3', lifecycle_stage: 'active' }),
  2: Fixtures.createExperiment({ experiment_id: '2', name: '2', lifecycle_stage: 'active' }),
};

describe('HomeView', () => {
  let wrapper;
  let minimalStore: any;
  let minimalProps: any;

  const mockStore = configureStore([thunk, promiseMiddleware()]);

  beforeEach(() => {
    minimalProps = {
      experimentIds: undefined,
      compareExperiments: false,
    };
    minimalStore = mockStore({
      entities: {
        experimentsById: {},
      },
      apis: jest.fn(() => ({})),
    });
  });

  test('should render with minimal props without exploding', () => {
    wrapper = mount(
      <Provider store={minimalStore}>
        <MemoryRouter>
          <HomeView />
        </MemoryRouter>
      </Provider>,
    );
    expect(wrapper.length).toBe(1);
  });

  test('getFirstActiveExperiment works', () => {
    expect(getFirstActiveExperiment(Object.values(experiments)).experiment_id).toEqual('2');
  });

  test('reroute to first active experiment works', () => {
    minimalStore = mockStore({
      entities: { experimentsById: experiments },
      apis: jest.fn(() => ({})),
    });
    wrapper = mount(
      <Provider store={minimalStore}>
        <MemoryRouter>
          <HomeView {...minimalProps} />
        </MemoryRouter>
      </Provider>,
    );

    expect(mockNavigate).toBeCalledWith({ replace: true, to: '/experiments/2' });
  });
});
