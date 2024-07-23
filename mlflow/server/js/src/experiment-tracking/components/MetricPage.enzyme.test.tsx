/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import { Provider } from 'react-redux';
import { MemoryRouter } from '../../common/utils/RoutingUtils';
import { mount } from 'enzyme';
import configureStore from 'redux-mock-store';

import { MetricPageImpl, MetricPage } from './MetricPage';
import NotFoundPage from './NotFoundPage';

describe('MetricPage', () => {
  let wrapper;
  let instance;
  let minimalProps: any;
  let commonProps: any;
  let minimalStore: any;

  const mockStore = configureStore([thunk, promiseMiddleware()]);

  beforeEach(() => {
    // TODO: remove global fetch mock by explicitly mocking all the service API calls
    // @ts-expect-error TS(2322): Type 'Mock<Promise<{ ok: true; status: number; tex... Remove this comment to see the full error message
    global.fetch = jest.fn(() => Promise.resolve({ ok: true, status: 200, text: () => Promise.resolve('') }));
    minimalProps = {
      dispatch: jest.fn(),
      location: {
        search: '?runs=[]&metric="metricKey"',
      },
    };
    commonProps = {
      ...minimalProps,
      experimentIds: [0],
    };

    minimalStore = mockStore({
      entities: {},
      apis: jest.fn((key) => {
        return {};
      }),
    });
  });

  test('should render with minimal props without exploding', () => {
    wrapper = mount(
      <Provider store={minimalStore}>
        <MemoryRouter>
          <MetricPage {...minimalProps} />
        </MemoryRouter>
      </Provider>,
    ).find(MetricPage);
    expect(wrapper.length).toBe(1);
  });

  test('should render NotFoundPage when runs are not in query parameters', () => {
    wrapper = mount(
      <Provider store={minimalStore}>
        <MemoryRouter>
          <MetricPage {...commonProps} />
        </MemoryRouter>
      </Provider>,
    ).find(MetricPage);

    instance = wrapper.find(MetricPageImpl).instance();
    expect(instance.renderPageContent().type).toBe(NotFoundPage);
  });

  test('should not render NotFoundPage when runs are provided in query parameters', () => {
    const props = {
      ...commonProps,
      location: {
        search:
          '?runs=["a"]&metric="primary_metric_key"&experiment=0&plot_metric_keys=["metric_1","metric_2"]&plot_layout={}',
      },
    };
    wrapper = mount(
      <Provider store={minimalStore}>
        <MemoryRouter>
          <MetricPage {...props} />
        </MemoryRouter>
      </Provider>,
    ).find(MetricPage);

    instance = wrapper.find(MetricPageImpl).instance();
    expect(instance.renderPageContent().type).not.toBe(NotFoundPage);
  });
});
