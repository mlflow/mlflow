import React from 'react';
import { ModelVersionView, ModelVersionViewImpl } from './ModelVersionView';
import { mockModelVersionDetailed, Stages, ACTIVE_STAGES, stageTagComponents, modelStageNames } from '../test-utils';
import { ModelVersionStatus } from '../constants';
import { BrowserRouter } from 'react-router-dom';
import Utils from '../../common/utils/Utils';
import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import { ModelVersionTag } from '../sdk/ModelRegistryMessages';
import { Provider } from 'react-redux';
import { mockRunInfo } from '../../experiment-tracking/utils/test-utils/ReduxStoreFixtures';
import Routers from '../../experiment-tracking/routes';
import { mountWithIntl } from '../../common/utils/TestUtils';

describe('ModelVersionView', () => {
  let wrapper;
  let minimalProps;
  let minimalStoreRaw;
  let minimalStore;
  const mockStore = configureStore([thunk, promiseMiddleware()]);

  beforeEach(() => {
    // TODO: remove global fetch mock by explicitly mocking all the service API calls
    global.fetch = jest.fn(() =>
      Promise.resolve({ ok: true, status: 200, text: () => Promise.resolve('') }),
    );
    minimalProps = {
      modelName: 'Model A',
      modelVersion: mockModelVersionDetailed(
        'Model A',
        1,
        Stages.PRODUCTION,
        ModelVersionStatus.READY,
      ),
      handleStageTransitionDropdownSelect: jest.fn(),
      deleteModelVersionApi: jest.fn(() => Promise.resolve()),
      handleEditDescription: jest.fn(() => Promise.resolve()),
      listModelStagesApi: jest.fn(),
      history: { push: jest.fn() },
      tags: {},
      schema: {
        inputs: [],
        outputs: [],
      },
      stageTagComponents: stageTagComponents(),
      modelStageNames: modelStageNames,
    };
    minimalStoreRaw = {
      entities: {
        tagsByModelVersion: {
          'Model A': {
            1: {
              'special key': ModelVersionTag.fromJs({
                key: 'special key',
                value: 'not so special value',
              }),
            },
          },
        },
        listModelStages: {
          'stageTagComponents': stageTagComponents(),
          'modelStageNames': modelStageNames
        }
      },
      apis: {},
    };
    minimalStore = mockStore(minimalStoreRaw);
  });

  test('should render with minimal props without exploding', () => {
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ModelVersionView {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    );
    expect(wrapper.length).toBe(1);
  });

  test('should render delete dropdown item when model version is ready', () => {
    const props = {
      ...minimalProps,
      modelVersion: mockModelVersionDetailed('Model A', 1, Stages.NONE, ModelVersionStatus.READY),
    };
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ModelVersionView {...props} />
        </BrowserRouter>
      </Provider>,
    );
    expect(wrapper.find('button[data-test-id="overflow-menu-trigger"]').length).toBe(1);
  });

  test('should disable dropdown delete menu item when model version is in active stage', () => {
    let i;
    for (i = 0; i < ACTIVE_STAGES.length; ++i) {
      const props = {
        ...minimalProps,
        modelVersion: mockModelVersionDetailed(
          'Model A',
          1,
          ACTIVE_STAGES[i],
          ModelVersionStatus.READY,
        ),
      };
      wrapper = mountWithIntl(
        <Provider store={minimalStore}>
          <BrowserRouter>
            <ModelVersionView {...props} />
          </BrowserRouter>
        </Provider>,
      );
      wrapper
        .find("[data-test-id='overflow-menu-trigger']")
        .at(0)
        .simulate('click');
      // The antd `Menu.Item` component converts the `disabled` attribute to `aria-disabled`
      // when generating HTML. Accordingly, we check for the presence of the `aria-disabled`
      // attribute within the rendered HTML.
      const deleteMenuItem = wrapper.find('[data-test-id="delete"]').hostNodes();
      expect(deleteMenuItem.prop('aria-disabled')).toBe(true);
      deleteMenuItem.simulate('click');
      expect(wrapper.find(ModelVersionViewImpl).instance().state.isDeleteModalVisible).toBe(false);
    }
  });

  test('should enable dropdown delete menu item when model version is in active stage', () => {
    const inactiveStages = [Stages.NONE, Stages.ARCHIVED];
    let i;
    for (i = 0; i < inactiveStages.length; ++i) {
      const props = {
        ...minimalProps,
        modelVersion: mockModelVersionDetailed(
          'Model A',
          1,
          inactiveStages[i],
          ModelVersionStatus.READY,
        ),
      };
      wrapper = mountWithIntl(
        <Provider store={minimalStore}>
          <BrowserRouter>
            <ModelVersionView {...props} />
          </BrowserRouter>
        </Provider>,
      );
      wrapper
        .find('button[data-test-id="overflow-menu-trigger"]')
        .at(0)
        .simulate('click');
      // The antd `Menu.Item` component converts the `disabled` attribute to `aria-disabled`
      // when generating HTML. Accordingly, we check for the presence of the `aria-disabled`
      // attribute within the rendered HTML.
      const deleteMenuItem = wrapper.find('[data-test-id="delete"]').hostNodes();
      expect(deleteMenuItem.prop('aria-disabled')).toBeFalsy();
      deleteMenuItem.simulate('click');
      expect(wrapper.find(ModelVersionViewImpl).instance().state.isDeleteModalVisible).toBe(true);
    }
  });

  test('run link renders if set', () => {
    const runLink =
      'https://other.mlflow.hosted.instance.com/experiments/18722387/' +
      'runs/d2c09dbd056c4d9c9289b854f491be10';
    const modelVersion = mockModelVersionDetailed(
      'Model A',
      1,
      Stages.NONE,
      ModelVersionStatus.READY,
      [],
      runLink,
    );
    const runId = 'somerunid';
    const experimentId = 'experiment_id';
    const runInfo = mockRunInfo(runId, experimentId);
    const expectedRunDisplayName = Utils.getRunDisplayName({}, runId);
    const props = {
      ...minimalProps,
      modelVersion: modelVersion,
      runInfo: runInfo,
      runDisplayName: expectedRunDisplayName,
    };
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ModelVersionView {...props} />
        </BrowserRouter>
      </Provider>,
    );
    const linkedRun = wrapper.find('.linked-run').at(0); // TODO: Figure out why it returns 2.
    expect(linkedRun.html()).toContain(runLink);
    expect(linkedRun.html()).toContain(runLink.substr(0, 37) + '...');
  });

  test('run name and link render if runinfo provided', () => {
    const runId = 'somerunid';
    const experimentId = 'experiment_id';
    const runInfo = mockRunInfo(runId, experimentId);
    const expectedRunLink = Routers.getRunPageRoute(experimentId, runId);
    const expectedRunDisplayName = Utils.getRunDisplayName({}, runId);
    const props = {
      ...minimalProps,
      runInfo: runInfo,
      runDisplayName: expectedRunDisplayName,
    };
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ModelVersionView {...props} />
        </BrowserRouter>
      </Provider>,
    );
    const linkedRun = wrapper.find('.linked-run').at(0); // TODO: Figure out why it returns 2.
    expect(linkedRun.html()).toContain(expectedRunLink);
    expect(linkedRun.html()).toContain(expectedRunDisplayName);
  });

  test('Page title is set', () => {
    const mockUpdatePageTitle = jest.fn();
    Utils.updatePageTitle = mockUpdatePageTitle;
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ModelVersionView {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    );
    expect(mockUpdatePageTitle.mock.calls[0][0]).toBe('Model A v1 - MLflow Model');
  });

  test('should tags rendered in the UI', () => {
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ModelVersionView {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    );
    expect(wrapper.html()).toContain('special key');
    expect(wrapper.html()).toContain('not so special value');
  });

  test('creator description not rendered if user_id is unavailable', () => {
    const props = {
      ...minimalProps,
      modelVersion: mockModelVersionDetailed(
        'Model A',
        1,
        Stages.NONE,
        ModelVersionStatus.READY,
        [],
        null,
        'b99a0fc567ae4d32994392c800c0b6ce',
        null,
      ),
    };
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ModelVersionView {...props} />
        </BrowserRouter>
      </Provider>,
    );

    expect(wrapper.find('.metadata-list td.ant-descriptions-item').length).toBe(4);
    expect(
      wrapper
        .find('.metadata-list span.ant-descriptions-item-label')
        .at(0)
        .text(),
    ).toBe('Registered At');
    expect(
      wrapper
        .find('.metadata-list span.ant-descriptions-item-label')
        .at(1)
        .text(),
    ).toBe('Stage');
    expect(
      wrapper
        .find('.metadata-list span.ant-descriptions-item-label')
        .at(2)
        .text(),
    ).toBe('Last Modified');
    expect(
      wrapper
        .find('.metadata-list span.ant-descriptions-item-label')
        .at(3)
        .text(),
    ).toBe('Source Run');
  });

  test('creator description rendered if user_id is available', () => {
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ModelVersionView {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    );

    expect(wrapper.find('.metadata-list td.ant-descriptions-item').length).toBe(5);
    expect(
      wrapper
        .find('.metadata-list span.ant-descriptions-item-label')
        .at(0)
        .text(),
    ).toBe('Registered At');
    expect(
      wrapper
        .find('.metadata-list span.ant-descriptions-item-label')
        .at(1)
        .text(),
    ).toBe('Creator');
    expect(
      wrapper
        .find('.metadata-list span.ant-descriptions-item-label')
        .at(2)
        .text(),
    ).toBe('Stage');
    expect(
      wrapper
        .find('.metadata-list span.ant-descriptions-item-label')
        .at(3)
        .text(),
    ).toBe('Last Modified');
    expect(
      wrapper
        .find('.metadata-list span.ant-descriptions-item-label')
        .at(4)
        .text(),
    ).toBe('Source Run');
  });
});
