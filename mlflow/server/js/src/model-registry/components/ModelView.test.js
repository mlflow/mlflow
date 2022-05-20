import React from 'react';
import { ModelView, ModelViewImpl, StageFilters } from './ModelView';
import { mockModelVersionDetailed, mockRegisteredModelDetailed, stageTagComponents, modelStageNames } from '../test-utils';
import { ModelVersionStatus } from '../constants';
import { Stages } from '../test-utils';
import { BrowserRouter } from 'react-router-dom';
import { ModelVersionTable } from './ModelVersionTable';
import Utils from '../../common/utils/Utils';
import { getCompareModelVersionsPageRoute } from '../routes';
import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import { RegisteredModelTag } from '../sdk/ModelRegistryMessages';
import { Provider } from 'react-redux';
import { mountWithIntl } from '../../common/utils/TestUtils';

describe('ModelView', () => {
  let wrapper;
  let instance;
  let minimalProps;
  let historyMock;
  let minimalStoreRaw;
  let minimalStore;
  const mockStore = configureStore([thunk, promiseMiddleware()]);

  const mockModel = {
    name: 'Model A',
    latestVersions: [
      mockModelVersionDetailed('Model A', 1, Stages.PRODUCTION, ModelVersionStatus.READY),
      mockModelVersionDetailed('Model A', 2, Stages.STAGING, ModelVersionStatus.READY),
      mockModelVersionDetailed('Model A', 3, Stages.NONE, ModelVersionStatus.READY),
    ],
    versions: [
      mockModelVersionDetailed('Model A', 1, Stages.PRODUCTION, ModelVersionStatus.READY),
      mockModelVersionDetailed('Model A', 2, Stages.STAGING, ModelVersionStatus.READY),
      mockModelVersionDetailed('Model A', 3, Stages.NONE, ModelVersionStatus.READY),
    ],
    tags: [
      {
        'special key': RegisteredModelTag.fromJs({
          key: 'special key',
          value: 'not so special value',
        }),
      },
    ],
  };

  beforeEach(() => {
    // TODO: remove global fetch mock by explicitly mocking all the service API calls
    global.fetch = jest.fn(() =>
      Promise.resolve({ ok: true, status: 200, text: () => Promise.resolve('') }),
    );
    historyMock = jest.fn();
    minimalProps = {
      model: mockRegisteredModelDetailed(
        mockModel.name,
        mockModel.latestVersions,
        mockModel.tags,
        mockModel.permissionLevel,
      ),
      modelVersions: mockModel.versions,
      handleEditDescription: jest.fn(),
      listModelStagesApi: jest.fn(),
      handleDelete: jest.fn(),
      showEditPermissionModal: jest.fn(),
      history: { push: historyMock },
      tags: {},
      setRegisteredModelTagApi: jest.fn(),
      deleteRegisteredModelTagApi: jest.fn(),
    };
    minimalStoreRaw = {
      entities: {
        tagsByRegisteredModel: {
          'Model A': {
            'special key': RegisteredModelTag.fromJs({
              key: 'special key',
              value: 'not so special value',
            }),
          },
        },
        listModelStages: {
          'stageTagComponents': stageTagComponents(),
          'modelStageNames': modelStageNames
        },
      },
      apis: {},
    };
    minimalStore = mockStore(minimalStoreRaw);
  });

  test('should render with minimal props without exploding', () => {
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ModelView {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    );
    expect(wrapper.find(ModelView).length).toBe(1);
  });

  test('should render all model versions initially', () => {
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ModelView {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    );
    expect(wrapper.find('td.model-version').length).toBe(3);
    expect(
      wrapper
        .find('td.model-version')
        .at(0)
        .text(),
    ).toBe('Version 3');
    expect(
      wrapper
        .find('td.model-version')
        .at(1)
        .text(),
    ).toBe('Version 2');
    expect(
      wrapper
        .find('td.model-version')
        .at(2)
        .text(),
    ).toBe('Version 1');
  });

  test('should render model version table with activeStageOnly when "Active" button is on', () => {
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ModelView {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    );
    expect(wrapper.find(ModelVersionTable).props().activeStageOnly).toBe(false);
    instance = wrapper.find(ModelViewImpl).instance();
    instance.setState({ stageFilter: StageFilters.ACTIVE });
    wrapper.update();
    expect(wrapper.find(ModelVersionTable).props().activeStageOnly).toBe(true);
  });

  test('Page title is set', () => {
    const mockUpdatePageTitle = jest.fn();
    Utils.updatePageTitle = mockUpdatePageTitle;
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ModelView {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    );
    expect(mockUpdatePageTitle.mock.calls[0][0]).toBe('Model A - MLflow Model');
  });

  test('should disable dropdown delete menu item when model has active versions', () => {
    const props = {
      ...minimalProps,
      model: {
        ...minimalProps.model,
      },
    };
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ModelView {...props} />
        </BrowserRouter>
      </Provider>,
    );
    wrapper.find('button[data-test-id="overflow-menu-trigger"]').simulate('click');
    // The antd `Menu.Item` component converts the `disabled` attribute to `aria-disabled`
    // when generating HTML. Accordingly, we check for the presence of the `aria-disabled`
    // attribute within the rendered HTML.
    const deleteMenuItem = wrapper.find('[data-test-id="delete"]').hostNodes();
    expect(deleteMenuItem.prop('aria-disabled')).toBe(true);
    deleteMenuItem.simulate('click');
    expect(wrapper.find(ModelViewImpl).instance().state.isDeleteModalVisible).toBe(false);
  });

  test('compare button is disabled when no/1 run selected, active when 2+ runs selected', () => {
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ModelView {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    );

    expect(wrapper.find('[data-test-id="compareButton"]').hostNodes().length).toBe(1);
    expect(
      wrapper
        .find('[data-test-id="compareButton"]')
        .hostNodes()
        .props().disabled,
    ).toEqual(true);

    wrapper
      .find(ModelViewImpl)
      .instance()
      .setState({
        runsSelected: { run_id_1: 'version_1' },
      });
    wrapper.update();
    expect(
      wrapper
        .find('[data-test-id="compareButton"]')
        .hostNodes()
        .props().disabled,
    ).toEqual(true);

    const twoRunsSelected = { run_id_1: 'version_1', run_id_2: 'version_2' };
    wrapper
      .find(ModelViewImpl)
      .instance()
      .setState({
        runsSelected: twoRunsSelected,
      });
    wrapper.update();
    expect(
      wrapper
        .find('[data-test-id="compareButton"]')
        .hostNodes()
        .props().disabled,
    ).toEqual(false);

    wrapper
      .find('[data-test-id="compareButton"]')
      .hostNodes()
      .simulate('click');
    expect(historyMock).toHaveBeenCalledWith(
      getCompareModelVersionsPageRoute(minimalProps['model']['name'], twoRunsSelected),
    );
  });

  test('should tags rendered in the UI', () => {
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ModelView {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    );
    expect(wrapper.html()).toContain('special key');
    expect(wrapper.html()).toContain('not so special value');
  });

  test('creator description not rendered if user_id is unavailable', () => {
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ModelView {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    );

    expect(wrapper.find('.metadata-list td.ant-descriptions-item').length).toBe(2);
    expect(
      wrapper
        .find('.metadata-list span.ant-descriptions-item-label')
        .at(0)
        .text(),
    ).toBe('Created Time');
    expect(
      wrapper
        .find('.metadata-list span.ant-descriptions-item-label')
        .at(1)
        .text(),
    ).toBe('Last Modified');
  });

  test('creator description rendered if user_id is available', () => {
    const user_id = 'email@databricks.com';
    const props = {
      ...minimalProps,
      model: {
        ...minimalProps.model,
        user_id,
      },
    };
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ModelView {...props} />
        </BrowserRouter>
      </Provider>,
    );

    expect(wrapper.find('.metadata-list td.ant-descriptions-item').length).toBe(3);
    expect(
      wrapper
        .find('.metadata-list span.ant-descriptions-item-label')
        .at(0)
        .text(),
    ).toBe('Created Time');
    expect(
      wrapper
        .find('.metadata-list span.ant-descriptions-item-label')
        .at(1)
        .text(),
    ).toBe('Last Modified');
    expect(
      wrapper
        .find('.metadata-list span.ant-descriptions-item-label')
        .at(2)
        .text(),
    ).toBe('Creator');

    expect(
      wrapper
        .find('.metadata-list span.ant-descriptions-item-content')
        .at(2)
        .text(),
    ).toBe(user_id);
  });
});
