import React from 'react';
import thunk from 'redux-thunk';
import configureStore from 'redux-mock-store';
import promiseMiddleware from 'redux-promise-middleware';
import { Provider } from 'react-redux';
import { BrowserRouter } from 'react-router-dom';
import Fixtures from '../utils/test-utils/Fixtures';
import HomeView, { getFirstActiveExperiment } from './HomeView';
import { mount } from 'enzyme';

const experiments = {
  1: Fixtures.createExperiment({ experiment_id: '1', name: '1', lifecycle_stage: 'deleted' }),
  3: Fixtures.createExperiment({ experiment_id: '3', name: '3', lifecycle_stage: 'active' }),
  2: Fixtures.createExperiment({ experiment_id: '2', name: '2', lifecycle_stage: 'active' }),
};

describe('HomeView', () => {
  let wrapper;
  let minimalStore;
  let minimalProps;
  let mockHistory;

  const mockStore = configureStore([thunk, promiseMiddleware()]);

  beforeEach(() => {
    mockHistory = { push: jest.fn() };
    minimalProps = {
      history: mockHistory,
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
        <BrowserRouter>
          <HomeView />
        </BrowserRouter>
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
        <BrowserRouter>
          <HomeView {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    );
    expect(mockHistory.push).toHaveBeenCalledWith('/experiments/2');
  });
});
