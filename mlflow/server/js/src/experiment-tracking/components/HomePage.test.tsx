/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

/* eslint react/prop-types:0 */
import React from 'react';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import { Provider } from 'react-redux';
import { MemoryRouter } from '../../common/utils/RoutingUtils';
import { shallow } from 'enzyme';
import configureStore from 'redux-mock-store';
import { HomePageImpl } from './HomePage';
import HomeView from './HomeView';

describe('HomePage', () => {
  let wrapper;
  let minimalProps: any;
  let minimalStore: any;

  const mockStore = configureStore([thunk, promiseMiddleware()]);

  beforeEach(() => {
    minimalProps = {
      history: {},
      dispatchSearchExperimentsApi: jest.fn(),
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
      wrappingComponent: (props: any) => {
        const { children } = props;
        return (
          <Provider store={minimalStore}>
            <MemoryRouter>{children}</MemoryRouter>
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
      wrappingComponent: (wrappingProps: any) => {
        const { children } = wrappingProps;
        return (
          <Provider store={minimalStore}>
            <MemoryRouter>{children}</MemoryRouter>
          </Provider>
        );
      },
    });
    expect(wrapper.find(HomeView).length).toBe(1);
  });
});
