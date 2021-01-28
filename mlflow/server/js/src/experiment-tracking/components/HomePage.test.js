/* eslint react/prop-types:0 */
import React from 'react';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import { Provider } from 'react-redux';
import { BrowserRouter } from 'react-router-dom';
import { shallow } from 'enzyme';
import configureStore from 'redux-mock-store';
import { HomePageImpl } from './HomePage';
import HomeView from './HomeView';

describe('HomePage', () => {
  let wrapper;
  let minimalProps;
  let minimalStore;

  const mockStore = configureStore([thunk, promiseMiddleware()]);

  beforeEach(() => {
    minimalProps = {
      dispatchListExperimentsApi: jest.fn(),
    };
    minimalStore = mockStore({
      entities: {},
      apis: jest.fn((key) => {
        return {};
      }),
    });
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<HomePageImpl {...minimalProps} />, {
      wrappingComponent: (props) => {
        const { children } = props;
        return (
          <Provider store={minimalStore}>
            <BrowserRouter>{children}</BrowserRouter>
          </Provider>
        );
      },
    });
    expect(wrapper.length).toBe(1);
  });

  test('should render HomeView', () => {
    const props = {
      ...minimalProps,
      experimentId: '0',
    };

    wrapper = shallow(<HomePageImpl {...props} />, {
      wrappingComponent: () => {
        const { children } = props;
        return (
          <Provider store={minimalStore}>
            <BrowserRouter>{children}</BrowserRouter>
          </Provider>
        );
      },
    });
    expect(wrapper.find(HomeView).length).toBe(1);
  });
});
