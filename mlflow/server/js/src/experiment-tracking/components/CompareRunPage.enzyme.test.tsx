/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { MemoryRouter } from '../../common/utils/RoutingUtils';
import CompareRunPage from './CompareRunPage';
import { Provider } from 'react-redux';
import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';

import { mountWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.enzyme';

describe('CompareRunPage', () => {
  let wrapper;
  let minimalProps: any;
  let minimalStore: any;
  const mockStore = configureStore([thunk, promiseMiddleware()]);

  beforeEach(() => {
    // TODO: remove global fetch mock by explicitly mocking all the service API calls
    // @ts-expect-error TS(2322): Type 'Mock<Promise<{ ok: true; status: number; tex... Remove this comment to see the full error message
    global.fetch = jest.fn(() => Promise.resolve({ ok: true, status: 200, text: () => Promise.resolve('') }));
    minimalProps = {
      location: {
        search: {
          '?runs': '["runn-1234-5678-9012", "runn-1234-5678-9034"]',
          experiments: '["12345"]',
        },
      },
      experimentIds: ['12345'],
      runUuids: ['runn-1234-5678-9012', 'runn-1234-5678-9034'],
      dispatch: jest.fn(),
    };
    minimalStore = mockStore({
      entities: {},
      apis: jest.fn((key) => {
        return {};
      }),
    });
  });

  test('should render with minimal props without exploding', () => {
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <MemoryRouter>
          <CompareRunPage {...minimalProps} />
        </MemoryRouter>
      </Provider>,
    );
    expect(wrapper.find(CompareRunPage).length).toBe(1);
  });
});

describe('CompareRunPage URI encoded', () => {
  let wrapper;
  let minimalProps: any;
  let minimalStore: any;
  const mockStore = configureStore([thunk, promiseMiddleware()]);

  beforeEach(() => {
    // TODO: remove global fetch mock by explicitly mocking all the service API calls
    // @ts-expect-error TS(2322): Type 'Mock<Promise<{ ok: true; status: number; tex... Remove this comment to see the full error message
    global.fetch = jest.fn(() => Promise.resolve({ ok: true, status: 200, text: () => Promise.resolve('') }));
    minimalProps = {
      location: {
        search:
          '?runs=%5B%252281d708375e574d6cbf4985b8701d67d2%2522,%25225f70fea1ef004d3180a6c34fe2d0d94e%2522%5D&experiments=%5B%25220%2522%5D',
      },
      experimentIds: ['12345'],
      runUuids: ['runn-1234-5678-9012', 'runn-1234-5678-9034'],
      dispatch: jest.fn(),
    };
    minimalStore = mockStore({
      entities: {},
      apis: jest.fn((key) => {
        return {};
      }),
    });
  });

  test('should render with minimal props without exploding', () => {
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <MemoryRouter>
          <CompareRunPage {...minimalProps} />
        </MemoryRouter>
      </Provider>,
    );
    expect(wrapper.find(CompareRunPage).length).toBe(1);
  });
});
