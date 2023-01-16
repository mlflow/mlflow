import React from 'react';
import { mount } from 'enzyme';
import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import { mockModelVersionDetailed, mockRegisteredModelDetailed } from '../test-utils';
import {
  ModelVersionStatus,
  REGISTERED_MODELS_SEARCH_TIMESTAMP_FIELD,
  REGISTERED_MODELS_SEARCH_NAME_FIELD,
  Stages,
} from '../constants';
import { Provider } from 'react-redux';
import { BrowserRouter } from 'react-router-dom';
import { ModelListPageImpl } from './ModelListPage';

describe('ModelListPage', () => {
  let wrapper;
  let minimalProps;
  let minimalStore;
  let instance;
  let pushSpy;
  const mockStore = configureStore([thunk, promiseMiddleware()]);
  const noop = () => {};
  const loadPageMock = (page, callback, errorCallback, isInitialLoading) => {};

  beforeEach(() => {
    const location = {};

    pushSpy = jest.fn();
    const history = {
      location: {
        pathName: '/models',
        search: '',
      },
      push: pushSpy,
    };
    minimalProps = {
      models: [],
      searchRegisteredModelsApi: jest.fn(() => Promise.resolve({})),
      listEndpointsApi: jest.fn(() => Promise.resolve({})),
      listEndpointsV2Api: jest.fn(() => Promise.resolve({})),
      getRegistryWidePermissionsApi: jest.fn(() => Promise.resolve({})),
      apis: {},
      history,
      location,
    };
    const name = 'Model A';
    const versions = [
      mockModelVersionDetailed('Model A', 1, Stages.PRODUCTION, ModelVersionStatus.READY),
    ];
    minimalStore = mockStore({
      entities: {
        modelByName: {
          [name]: mockRegisteredModelDetailed(name, versions),
        },
      },
      apis: {},
    });
  });

  describe('the states should be correctly set upon user input and clear', () => {
    beforeEach(() => {
      wrapper = mount(
        <Provider store={minimalStore}>
          <BrowserRouter>
            <ModelListPageImpl {...minimalProps} />
          </BrowserRouter>
        </Provider>,
      );
    });

    test('the states should be correctly set when page is loaded initially', () => {
      instance = wrapper.find(ModelListPageImpl).instance();
      jest.spyOn(instance, 'loadPage').mockImplementation(loadPageMock);
      expect(instance.state.searchInput).toBe('');
      expect(instance.state.orderByKey).toBe(REGISTERED_MODELS_SEARCH_NAME_FIELD);
      expect(instance.state.orderByAsc).toBe(true);
    });

    test('the states should be correctly set when user enters name search', () => {
      instance = wrapper.find(ModelListPageImpl).instance();
      jest.spyOn(instance, 'loadPage').mockImplementation(loadPageMock);
      instance.handleSearch(noop, noop, 'abc');
      expect(instance.state.searchInput).toBe('abc');
      expect(instance.state.currentPage).toBe(1);
    });

    test('the states should be correctly set when user enters tag search', () => {
      instance = wrapper.find(ModelListPageImpl).instance();
      jest.spyOn(instance, 'loadPage').mockImplementation(loadPageMock);
      instance.handleSearch(noop, noop, 'tags.k = v');
      expect(instance.state.searchInput).toBe('tags.k = v');
      expect(instance.state.currentPage).toBe(1);
    });

    test('the states should be correctly set when user enters name and tag search', () => {
      instance = wrapper.find(ModelListPageImpl).instance();
      jest.spyOn(instance, 'loadPage').mockImplementation(loadPageMock);
      instance.handleSearch(noop, noop, 'name ilike "%xyz%" AND tags.k="v"');
      expect(instance.state.searchInput).toBe('name ilike "%xyz%" AND tags.k="v"');
      expect(instance.state.currentPage).toBe(1);
    });
    test('the states should be correctly set when user clear input', () => {
      instance = wrapper.find(ModelListPageImpl).instance();
      instance.state.searchInput = 'abc';
      jest.spyOn(instance, 'loadPage').mockImplementation(loadPageMock);
      instance.handleClear(noop, noop);
      expect(instance.state.currentPage).toBe(1);
      expect(instance.state.searchInput).toBe('');
    });
  });

  test('should render with minimal props and store without exploding', () => {
    wrapper = mount(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ModelListPageImpl {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    );
    expect(wrapper.find(ModelListPageImpl).length).toBe(1);
  });

  test('updateUrlWithSearchFilter correctly pushes url with params', () => {
    wrapper = mount(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ModelListPageImpl {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    );
    instance = wrapper.find(ModelListPageImpl).instance();
    instance.updateUrlWithSearchFilter(
      'name ilike "%name%" AND tag.key=value',
      REGISTERED_MODELS_SEARCH_TIMESTAMP_FIELD,
      false,
      2,
    );
    const expectedUrl = `/models?searchInput=name%20ilike%20%22%25name%25%22%20AND%20tag.key%3Dvalue&orderByKey=timestamp&orderByAsc=false&page=2`;
    expect(pushSpy).toHaveBeenCalledWith(expectedUrl);
  });

  test('should construct pushes URL correctly from old URLs with nameSearchInput', () => {
    minimalProps['location']['search'] = 'nameSearchInput=abc';
    wrapper = mount(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ModelListPageImpl {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    );
    instance = wrapper.find(ModelListPageImpl).instance();
    const expectedUrl = '/models?searchInput=abc';
    instance.render();
    expect(pushSpy).toHaveBeenCalledWith(expectedUrl);
  });

  test('should pushes URL correctly from old URLs with tagSearchInput', () => {
    minimalProps['location']['search'] = 'tagSearchInput=tags.k%20%3D%20"v"';
    wrapper = mount(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ModelListPageImpl {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    );
    instance = wrapper.find(ModelListPageImpl).instance();
    const expectedUrl = `/models?searchInput=tags.k%20%3D%20%22v%22`;
    instance.render();
    expect(pushSpy).toHaveBeenCalledWith(expectedUrl);
  });

  test('should pushes URL correctly from old URLs with nameSearchInput and tagSearchInput', () => {
    minimalProps['location']['search'] = 'nameSearchInput=abc&tagSearchInput=tags.k%20%3D%20"v"';
    wrapper = mount(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ModelListPageImpl {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    );
    instance = wrapper.find(ModelListPageImpl).instance();
    const expectedUrl =
      '/models?searchInput=name%20ilike%20%27%25abc%25%27%20AND%20tags.k%20%3D%20%22v%22';
    instance.render();
    expect(pushSpy).toHaveBeenCalledWith(expectedUrl);
  });

  test('should pushes URL correctly from URLs with searchInput', () => {
    minimalProps['location']['search'] =
      'searchInput=name%20ilike%20"%25ab%25"%20AND%20tags.a%20%3D%201';
    wrapper = mount(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ModelListPageImpl {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    );
    instance = wrapper.find(ModelListPageImpl).instance();
    const expectedUrl =
      '/models?searchInput=name%20ilike%20%22%25ab%25%22%20AND%20tags.a%20%3D%201';
    instance.render();
    expect(pushSpy).toHaveBeenCalledWith(expectedUrl);
  });
  // eslint-disable-next-line
});
