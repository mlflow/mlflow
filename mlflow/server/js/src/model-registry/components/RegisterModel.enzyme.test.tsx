/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import { jest, describe, beforeEach, test, expect } from '@jest/globals';
import React from 'react';
import { shallowWithInjectIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.enzyme';
import { mountWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.enzyme';
import configureStore from 'redux-mock-store';
import promiseMiddleware from 'redux-promise-middleware';
import thunk from 'redux-thunk';
import { RegisterModelWithIntl } from './RegisterModel';
import { getProtoField } from '../utils';
describe('RegisterModelButton', () => {
  let wrapper;
  let minimalProps: any;
  let minimalStore: any;
  const mockStore = configureStore([thunk, promiseMiddleware()]);

  beforeEach(() => {
    minimalProps = {
      disabled: false,
      runUuid: 'runUuid',
      modelPath: 'modelPath',
      // connected props
      modelByName: {},
      createRegisteredModelApi: jest.fn(() => Promise.resolve({})),
      createModelVersionApi: jest.fn(() => Promise.resolve({})),
      searchModelVersionsApi: jest.fn(() => Promise.resolve({})),
      searchRegisteredModelsApi: jest.fn(() => Promise.resolve({})),
    };
    minimalStore = mockStore({
      entities: {
        modelByName: {},
      },
    });
  });

  test('should render with minimal props and store without exploding', () => {
    wrapper = mountWithIntl(<RegisterModelWithIntl {...minimalProps} store={minimalStore} />);
    expect(wrapper.find('button').length).toBe(1);
  });

  test('handleSearchRegisteredModel should invoke api', () => {
    const response = { value: {} };
    // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
    response.value[getProtoField('registered_models')] = [
      {
        name: 'Model A',
      },
    ];

    const searchRegisteredModelsApi = jest.fn(() => Promise.resolve(response));
    const props = {
      ...minimalProps,
      searchRegisteredModelsApi,
    };
    wrapper = shallowWithInjectIntl(<RegisterModelWithIntl {...props} store={minimalStore} />);
    expect(props.searchRegisteredModelsApi.mock.calls.length).toBe(1);
    const instance = wrapper.instance();
    instance.handleSearchRegisteredModels('A');
    expect(props.searchRegisteredModelsApi.mock.calls.length).toBe(2);
  });

  describe('source URI construction', () => {
    test('should use models:/ format for logged models with model_id', async () => {
      const createModelVersionApi = jest.fn((...params: any) => Promise.resolve({}));
      const props = {
        ...minimalProps,
        runUuid: 'test-run-uuid',
        modelPath: 'mlflow-artifacts:/exp/models/m-12345/artifacts',
        modelRelativePath: 'model',
        loggedModelId: 'm-12345-model-id',
        createModelVersionApi,
      };
      wrapper = mountWithIntl(<RegisterModelWithIntl {...props} store={minimalStore} />);
      const instance = wrapper.find('RegisterModelImpl').instance();

      // Mock form validation
      instance.form.current = {
        validateFields: jest.fn(() =>
          Promise.resolve({
            selectedModel: 'existing-model',
          }),
        ),
        resetFields: jest.fn(),
      };

      await instance.handleRegisterModel();

      expect(createModelVersionApi).toHaveBeenCalledWith(
        'existing-model',
        'models:/m-12345-model-id', // Should use models:/ format
        'test-run-uuid',
        [],
        expect.any(String),
        'm-12345-model-id',
      );
    });

    test('should use runs:/ format for regular artifacts with run context', async () => {
      const createModelVersionApi = jest.fn((...params: any) => Promise.resolve({}));
      const props = {
        ...minimalProps,
        runUuid: 'test-run-uuid',
        modelPath: 'file:///path/to/artifacts/my_model',
        modelRelativePath: 'my_model',
        createModelVersionApi,
      };
      wrapper = mountWithIntl(<RegisterModelWithIntl {...props} store={minimalStore} />);
      const instance = wrapper.find('RegisterModelImpl').instance();

      // Mock form validation
      instance.form.current = {
        validateFields: jest.fn(() =>
          Promise.resolve({
            selectedModel: 'existing-model',
          }),
        ),
        resetFields: jest.fn(),
      };

      await instance.handleRegisterModel();

      expect(createModelVersionApi).toHaveBeenCalledWith(
        'existing-model',
        'runs:/test-run-uuid/my_model', // Should use runs:/ format
        'test-run-uuid',
        [],
        expect.any(String),
        undefined,
      );
    });

    test('should fall back to absolute modelPath when no run context', async () => {
      const createModelVersionApi = jest.fn((...params: any) => Promise.resolve({}));
      const props = {
        ...minimalProps,
        modelPath: 'file:///absolute/path/to/model',
        runUuid: undefined,
        modelRelativePath: undefined,
        createModelVersionApi,
      };
      wrapper = mountWithIntl(<RegisterModelWithIntl {...props} store={minimalStore} />);
      const instance = wrapper.find('RegisterModelImpl').instance();

      // Mock form validation
      instance.form.current = {
        validateFields: jest.fn(() =>
          Promise.resolve({
            selectedModel: 'existing-model',
          }),
        ),
        resetFields: jest.fn(),
      };

      await instance.handleRegisterModel();

      expect(createModelVersionApi).toHaveBeenCalledWith(
        'existing-model',
        'file:///absolute/path/to/model', // Should use absolute path as fallback
        undefined,
        [],
        expect.any(String),
        undefined,
      );
    });
  });
});
