import React from 'react';
import { mount } from 'enzyme';
import { ModelVersionView, ModelVersionViewImpl } from './ModelVersionView';
import { mockModelVersionDetailed } from '../test-utils';
import { Stages, ModelVersionStatus, ACTIVE_STAGES } from '../constants';
import { Dropdown, Tooltip } from 'antd';
import { BrowserRouter } from 'react-router-dom';
import Utils from '../../common/utils/Utils';
import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import { ModelVersionTag } from '../sdk/ModelRegistryMessages';
import { Provider } from 'react-redux';
import { mockRunInfo } from '../../experiment-tracking/utils/test-utils/ReduxStoreFixtures';
import Routers from '../../experiment-tracking/routes';

describe('ModelVersionView', () => {
  let wrapper;
  let minimalProps;
  let minimalStoreRaw;
  let minimalStore;
  const mockStore = configureStore([thunk, promiseMiddleware()]);

  beforeEach(() => {
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
      setModelVersionTagApi: jest.fn(),
      deleteModelVersionTagApi: jest.fn(),
      history: { push: jest.fn() },
      tags: {},
      schema: {
        inputs: [],
        outputs: [],
      },
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
      },
      apis: {},
    };
    minimalStore = mockStore(minimalStoreRaw);
  });

  test('should render with minimal props without exploding', () => {
    wrapper = mount(
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
    wrapper = mount(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ModelVersionView {...props} />
        </BrowserRouter>
      </Provider>,
    );
    expect(wrapper.find('.breadcrumb-header').find(Dropdown).length).toBe(1);
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
      wrapper = mount(
        <Provider store={minimalStore}>
          <BrowserRouter>
            <ModelVersionView {...props} />
          </BrowserRouter>
        </Provider>,
      );
      wrapper
        .find('.breadcrumb-dropdown')
        .hostNodes()
        .simulate('click');
      // The antd `Menu.Item` component converts the `disabled` attribute to `aria-disabled`
      // when generating HTML. Accordingly, we check for the presence of the `aria-disabled`
      // attribute within the rendered HTML.
      const deleteMenuItem = wrapper.find('.delete').hostNodes();
      expect(deleteMenuItem.prop('aria-disabled')).toBe(true);
      deleteMenuItem.simulate('click');
      expect(wrapper.find(ModelVersionViewImpl).instance().state.isDeleteModalVisible).toBe(false);
    }
  });

  test('should place tooltip on the right', () => {
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
      wrapper = mount(
        <Provider store={minimalStore}>
          <BrowserRouter>
            <ModelVersionView {...props} />
          </BrowserRouter>
        </Provider>,
      );
      wrapper
        .find('.breadcrumb-dropdown')
        .hostNodes()
        .simulate('click');
      const deleteMenuItem = wrapper.find('.delete').hostNodes();
      const tooltip = deleteMenuItem.find(Tooltip);
      expect(tooltip.prop('placement')).toBe('right');
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
      wrapper = mount(
        <Provider store={minimalStore}>
          <BrowserRouter>
            <ModelVersionView {...props} />
          </BrowserRouter>
        </Provider>,
      );
      wrapper
        .find('.breadcrumb-dropdown')
        .hostNodes()
        .simulate('click');
      // The antd `Menu.Item` component converts the `disabled` attribute to `aria-disabled`
      // when generating HTML. Accordingly, we check for the presence of the `aria-disabled`
      // attribute within the rendered HTML.
      const deleteMenuItem = wrapper.find('.delete').hostNodes();
      expect(deleteMenuItem.prop('aria-disabled')).toBeUndefined();
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
    wrapper = mount(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ModelVersionView {...props} />
        </BrowserRouter>
      </Provider>,
    );
    expect(wrapper.find('.linked-run').html()).toContain(runLink);
    expect(wrapper.find('.linked-run').html()).toContain(runLink.substr(0, 37) + '...');
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
    wrapper = mount(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ModelVersionView {...props} />
        </BrowserRouter>
      </Provider>,
    );
    expect(wrapper.find('.linked-run').html()).toContain(expectedRunLink);
    expect(wrapper.find('.linked-run').html()).toContain(expectedRunDisplayName);
  });

  test('Page title is set', () => {
    const mockUpdatePageTitle = jest.fn();
    Utils.updatePageTitle = mockUpdatePageTitle;
    wrapper = mount(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ModelVersionView {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    );
    expect(mockUpdatePageTitle.mock.calls[0][0]).toBe('Model A v1 - MLflow Model');
  });

  test('should tags rendered in the UI', () => {
    wrapper = mount(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ModelVersionView {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    );
    expect(wrapper.html()).toContain('special key');
    expect(wrapper.html()).toContain('not so special value');
  });
});
