/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

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
});
