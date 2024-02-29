/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import { mockModelVersionDetailed, mockRegisteredModelDetailed } from '../test-utils';
import { ModelVersionStatus, Stages } from '../constants';
import { Provider } from 'react-redux';
import { MemoryRouter } from '../../common/utils/RoutingUtils';
import { ModelPageImpl, ModelPage } from './ModelPage';
import Utils from '../../common/utils/Utils';
import { mountWithIntl } from 'common/utils/TestUtils.enzyme';
import { ModelRegistryRoutes } from '../routes';
import { ErrorWrapper } from '../../common/utils/ErrorWrapper';

describe('ModelPage', () => {
  let wrapper;
  let instance;
  let minimalProps: any;
  let minimalStore: any;
  const mockStore = configureStore([thunk, promiseMiddleware()]);
  const navigate = jest.fn();

  beforeEach(() => {
    minimalProps = {
      searchModelVersionsApi: jest.fn(() => Promise.resolve({})),
      getRegisteredModelDetailsApi: jest.fn(() => Promise.resolve({})),
      navigate,
    };
    const versions = [mockModelVersionDetailed('Model A', 1, Stages.PRODUCTION, ModelVersionStatus.READY)];
    minimalStore = mockStore({
      entities: {
        modelByName: {
          // @ts-expect-error TS(2345): Argument of type '{ name: any; creation_timestamp:... Remove this comment to see the full error message
          'Model A': mockRegisteredModelDetailed('Model A', versions),
        },
        modelVersionsByModel: {
          'Model A': {
            1: mockModelVersionDetailed('Model A', 1, Stages.PRODUCTION, ModelVersionStatus.READY),
          },
        },
      },
      apis: {},
    });
  });

  test('should render with minimal props and store without exploding', () => {
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <MemoryRouter>
          <ModelPage {...minimalProps} />
        </MemoryRouter>
      </Provider>,
    );
    expect(wrapper.find(ModelPage).length).toBe(1);
  });

  test('should redirect to model listing page when model is deleted', async () => {
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <MemoryRouter>
          <ModelPage {...minimalProps} />
        </MemoryRouter>
      </Provider>,
    );
    instance = wrapper.find(ModelPageImpl).instance();
    const mockError = new ErrorWrapper('{ "error_code": "RESOURCE_DOES_NOT_EXIST", "message": "Foo!" }', 404);

    Utils.isBrowserTabVisible = jest.fn(() => true);
    instance.loadData = jest.fn().mockReturnValue(Promise.reject(mockError));
    await instance.pollData();
    expect(navigate).toHaveBeenCalledWith(ModelRegistryRoutes.modelListPageRoute);
  });
});
