import React from 'react';
import { mount, shallow } from 'enzyme';
import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import { Provider } from 'react-redux';
import { BrowserRouter } from 'react-router-dom';
import { CompareModelVersionsPageImpl, CompareModelVersionsPage } from './CompareModelVersionsPage';
import { mockAjax } from '../../common/utils/TestUtils';

describe('CompareModelVersionPage', () => {
  let wrapper;
  let minimalStore;
  let minimalProps;
  const mockStore = configureStore([thunk, promiseMiddleware()]);

  beforeEach(() => {
    mockAjax();
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
        <BrowserRouter>
          <CompareModelVersionsPage {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    );
  });

  test('should render with name with model name with special characters', () => {
    const props = {
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
        <BrowserRouter>
          <CompareModelVersionsPage {...props} />
        </BrowserRouter>
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
    await expect(getRunApi).toBeCalled();
    await expect(getModelVersionArtifactApi).toBeCalled();
    expect(wrapper2.state('requestIds').length).toBe(2);
  });
});
