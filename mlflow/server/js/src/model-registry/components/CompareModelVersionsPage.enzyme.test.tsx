/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { mount, shallow } from 'enzyme';
import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import { Provider } from 'react-redux';
import { MemoryRouter } from '../../common/utils/RoutingUtils';
import { CompareModelVersionsPageImpl, CompareModelVersionsPage } from './CompareModelVersionsPage';

describe('CompareModelVersionPage', () => {
  let wrapper;
  let minimalStore: any;
  let minimalProps: any;
  const mockStore = configureStore([thunk, promiseMiddleware()]);

  beforeEach(() => {
    // TODO: remove global fetch mock by explicitly mocking all the service API calls
    // @ts-expect-error TS(2322): Type 'Mock<Promise<{ ok: true; status: number; tex... Remove this comment to see the full error message
    global.fetch = jest.fn(() => Promise.resolve({ ok: true, status: 200, text: () => Promise.resolve('') }));
    const modelName = 'normal-model-name';
    minimalProps = {
      location: {
        search:
          '?name=' +
          encodeURI(JSON.stringify(modelName)) +
          '&runs=' +
          JSON.stringify({
            1: '123',
            2: '234',
          }),
      },
      versionsToRuns: {
        1: '123',
        2: '234',
      },
      getRegisteredModelApi: jest.fn(),
      getModelVersionApi: jest.fn(),
      parseMlModelFile: jest.fn(),
    };
    minimalStore = mockStore({
      apis: {},
    });
  });

  test('should render with minimal props and store without exploding', () => {
    wrapper = mount(
      <Provider store={minimalStore}>
        <MemoryRouter>
          <CompareModelVersionsPage {...minimalProps} />
        </MemoryRouter>
      </Provider>,
    );
  });

  test('should render with name with model name with special characters', () => {
    const props: any = {
      location: {
        search:
          '?name=' +
          encodeURI(JSON.stringify('funky?!@#$^*()_=name~[]')) +
          '&runs=' +
          JSON.stringify({
            1: '123',
            2: '234',
          }),
      },
    };
    wrapper = mount(
      <Provider store={minimalStore}>
        <MemoryRouter>
          <CompareModelVersionsPage {...props} />
        </MemoryRouter>
      </Provider>,
    );
    expect(wrapper.find(CompareModelVersionsPageImpl).length).toBe(1);
  });

  test('should remove getRunRequest and getMlModelFileRequest api ids from state on 404', async () => {
    const mockError = {
      getErrorCode() {
        return 'RESOURCE_DOES_NOT_EXIST';
      },
    };
    const getRunApi = jest.fn().mockReturnValue(Promise.reject(mockError));
    const getModelVersionArtifactApi = jest.fn().mockReturnValue(Promise.reject(mockError));
    const myProps = {
      getRunApi: getRunApi,
      getModelVersionArtifactApi: getModelVersionArtifactApi,
      ...minimalProps,
    };
    const wrapper2 = shallow(<CompareModelVersionsPageImpl {...myProps} />);
    expect(wrapper2.state('requestIds').length).toBe(4);
    await expect(getRunApi).toHaveBeenCalled();
    await expect(getModelVersionArtifactApi).toHaveBeenCalled();
    expect(wrapper2.state('requestIds').length).toBe(2);
  });
});
