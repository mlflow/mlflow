/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { ModelVersionView, ModelVersionViewImpl } from './ModelVersionView';
import { mockModelVersionDetailed } from '../test-utils';
import { Stages, ModelVersionStatus, ACTIVE_STAGES } from '../constants';
import { MemoryRouter } from '../../common/utils/RoutingUtils';
import Utils from '../../common/utils/Utils';
import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import { ModelVersionTag } from '../sdk/ModelRegistryMessages';
import { Provider } from 'react-redux';
import { mockRunInfo } from '../../experiment-tracking/utils/test-utils/ReduxStoreFixtures';
import TrackingRouters from '../../experiment-tracking/routes';
import { ModelRegistryRoutes } from '../routes';
import { mountWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.enzyme';
import { DesignSystemContainer } from '../../common/components/DesignSystemContainer';
import { Services } from '../services';
import { shouldShowModelsNextUI } from '../../common/utils/FeatureUtils';

jest.spyOn(Services, 'searchRegisteredModels').mockResolvedValue({});
jest.mock('../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../common/utils/FeatureUtils')>('../../common/utils/FeatureUtils'),
  shouldShowModelsNextUI: jest.fn(),
}));
describe('ModelVersionView', () => {
  let wrapper;
  let minimalProps: any;
  let minimalStoreRaw;
  let minimalStore: any;
  let createComponentInstance: any;
  const mockStore = configureStore([thunk, promiseMiddleware()]);
  beforeEach(() => {
    minimalProps = {
      modelName: 'Model A',
      modelVersion: mockModelVersionDetailed('Model A', 1, Stages.PRODUCTION, ModelVersionStatus.READY),
      handleStageTransitionDropdownSelect: jest.fn(),
      deleteModelVersionApi: jest.fn(() => Promise.resolve()),
      handleEditDescription: jest.fn(() => Promise.resolve()),
      setModelVersionTagApi: jest.fn(),
      deleteModelVersionTagApi: jest.fn(),
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
              'special key': (ModelVersionTag as any).fromJs({
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
    createComponentInstance = (props: any) =>
      mountWithIntl(
        <DesignSystemContainer>
          <Provider store={minimalStore}>
            <MemoryRouter>
              <ModelVersionView {...props} />
            </MemoryRouter>
          </Provider>
        </DesignSystemContainer>,
      );
  });
  test('should render with minimal props without exploding', () => {
    wrapper = createComponentInstance(minimalProps);
    expect(wrapper.length).toBe(1);
  });
  test('should render delete dropdown item when model version is ready', () => {
    const props = {
      ...minimalProps,
      modelVersion: mockModelVersionDetailed('Model A', 1, Stages.NONE, ModelVersionStatus.READY),
    };
    wrapper = createComponentInstance(props);
    expect(wrapper.find('button[data-testid="overflow-menu-trigger"]').length).toBe(1);
  });
  test('should disable dropdown delete menu item when model version is in active stage', () => {
    let i;
    for (i = 0; i < ACTIVE_STAGES.length; ++i) {
      const props = {
        ...minimalProps,
        modelVersion: mockModelVersionDetailed('Model A', 1, ACTIVE_STAGES[i], ModelVersionStatus.READY),
      };
      wrapper = createComponentInstance(props);
      wrapper.find("button[data-testid='overflow-menu-trigger']").simulate('click');
      // The antd `Menu.Item` component converts the `disabled` attribute to `aria-disabled`
      // when generating HTML. Accordingly, we check for the presence of the `aria-disabled`
      // attribute within the rendered HTML.
      const deleteMenuItem = wrapper.find('[data-testid="delete"]').hostNodes();
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
        modelVersion: mockModelVersionDetailed('Model A', 1, inactiveStages[i], ModelVersionStatus.READY),
      };
      wrapper = createComponentInstance(props);
      wrapper.find('button[data-testid="overflow-menu-trigger"]').at(0).simulate('click');
      // The antd `Menu.Item` component converts the `disabled` attribute to `aria-disabled`
      // when generating HTML. Accordingly, we check for the presence of the `aria-disabled`
      // attribute within the rendered HTML.
      const deleteMenuItem = wrapper.find('[data-testid="delete"]').hostNodes();
      expect(deleteMenuItem.prop('aria-disabled')).toBeFalsy();
      deleteMenuItem.simulate('click');
      expect(wrapper.find(ModelVersionViewImpl).instance().state.isDeleteModalVisible).toBe(true);
    }
  });
  test('run link renders if set', () => {
    const runLink =
      'https://other.mlflow.hosted.instance.com/experiments/18722387/runs/d2c09dbd056c4d9c9289b854f491be10';
    const modelVersion = mockModelVersionDetailed(
      'Model A',
      1,
      Stages.NONE,
      ModelVersionStatus.READY,
      [],
      // @ts-expect-error TS(2345): Argument of type 'string' is not assignable to par... Remove this comment to see the full error message
      runLink,
    );
    const runId = 'somerunid';
    const experimentId = 'experiment_id';
    // @ts-expect-error TS(2345): Argument of type '"experiment_id"' is not assignab... Remove this comment to see the full error message
    const runInfo = mockRunInfo(runId, experimentId);
    const expectedRunDisplayName = Utils.getRunDisplayName({}, runId);
    const props = {
      ...minimalProps,
      modelVersion: modelVersion,
      runInfo: runInfo,
      runDisplayName: expectedRunDisplayName,
    };
    wrapper = createComponentInstance(props);
    const linkedRun = wrapper.find('.linked-run').at(0); // TODO: Figure out why it returns 2.
    expect(linkedRun.html()).toContain(runLink);
    expect(linkedRun.html()).toContain(runLink.substr(0, 37) + '...');
  });
  test('run name and link render if runinfo provided', () => {
    const runId = 'somerunid';
    const experimentId = 'experiment_id';
    // @ts-expect-error TS(2345): Argument of type '"experiment_id"' is not assignab... Remove this comment to see the full error message
    const runInfo = mockRunInfo(runId, experimentId);
    const expectedRunLink = TrackingRouters.getRunPageRoute(experimentId, runId);
    const expectedRunDisplayName = Utils.getRunDisplayName({}, runId);
    const props = {
      ...minimalProps,
      runInfo: runInfo,
      runDisplayName: expectedRunDisplayName,
    };
    wrapper = createComponentInstance(props);
    const linkedRun = wrapper.find('.linked-run').at(0); // TODO: Figure out why it returns 2.
    expect(linkedRun.html()).toContain(expectedRunLink);
    expect(linkedRun.html()).toContain(expectedRunDisplayName);
  });
  test('Page title is set', () => {
    const mockUpdatePageTitle = jest.fn();
    Utils.updatePageTitle = mockUpdatePageTitle;
    wrapper = createComponentInstance(minimalProps);
    expect(mockUpdatePageTitle.mock.calls[0][0]).toBe('Model A v1 - MLflow Model');
  });
  test('should tags rendered in the UI', () => {
    wrapper = createComponentInstance(minimalProps);
    expect(wrapper.html()).toContain('special key');
    expect(wrapper.html()).toContain('not so special value');
  });
  test('creator description not rendered if user_id is unavailable', () => {
    jest.mocked(shouldShowModelsNextUI).mockImplementation(() => false);
    const props = {
      ...minimalProps,
      modelVersion: mockModelVersionDetailed(
        'Model A',
        1,
        Stages.NONE,
        ModelVersionStatus.READY,
        [],
        // @ts-expect-error TS(2345): Argument of type 'null' is not assignable to param... Remove this comment to see the full error message
        null,
        'b99a0fc567ae4d32994392c800c0b6ce',
        null,
      ),
    };
    wrapper = createComponentInstance(props);
    expect(wrapper.find('[data-testid="descriptions-item"]').length).toBe(4);
    expect(wrapper.find('[data-testid="descriptions-item-label"]').at(0).text()).toBe('Registered At');
    expect(wrapper.find('[data-testid="descriptions-item-label"]').at(1).text()).toBe('Last Modified');
    expect(wrapper.find('[data-testid="descriptions-item-label"]').at(2).text()).toBe('Source Run');
    expect(wrapper.find('[data-testid="descriptions-item-label"]').at(3).text()).toBe('Stage');
  });
  test('creator description rendered if user_id is available', () => {
    jest.mocked(shouldShowModelsNextUI).mockImplementation(() => false);
    wrapper = createComponentInstance(minimalProps);
    expect(wrapper.find('[data-testid="descriptions-item"]').length).toBe(5);
    expect(wrapper.find('[data-testid="descriptions-item-label"]').at(0).text()).toBe('Registered At');
    expect(wrapper.find('[data-testid="descriptions-item-label"]').at(1).text()).toBe('Creator');
    expect(wrapper.find('[data-testid="descriptions-item-label"]').at(2).text()).toBe('Last Modified');
    expect(wrapper.find('[data-testid="descriptions-item-label"]').at(3).text()).toBe('Source Run');
    expect(wrapper.find('[data-testid="descriptions-item-label"]').at(4).text()).toBe('Stage');
  });
  test('should render copied from link when model version is a copy', () => {
    jest.mocked(shouldShowModelsNextUI).mockImplementation(() => true);
    const props = {
      ...minimalProps,
      modelVersion: mockModelVersionDetailed(
        'Model A',
        1,
        Stages.NONE,
        ModelVersionStatus.READY,
        [],
        undefined,
        'b99a0fc567ae4d32994392c800c0b6ce',
        'richard@example.com',
        'models:/Model B/2',
      ),
    };
    wrapper = createComponentInstance(props);
    expect(wrapper.find('[data-testid="descriptions-item"]').length).toBe(7);
    expect(wrapper.find('[data-testid="descriptions-item-label"]').at(0).text()).toBe('Registered At');
    expect(wrapper.find('[data-testid="descriptions-item-label"]').at(1).text()).toBe('Creator');
    expect(wrapper.find('[data-testid="descriptions-item-label"]').at(2).text()).toBe('Last Modified');
    expect(wrapper.find('[data-testid="descriptions-item-label"]').at(3).text()).toBe('Source Run');
    expect(wrapper.find('[data-testid="descriptions-item-label"]').at(4).text()).toBe('Copied from');
    expect(wrapper.find('[data-testid="descriptions-item-label"]').at(5).text()).toBe('Aliases');
    expect(wrapper.find('[data-testid="descriptions-item-label"]').at(6).text()).toBe('Stage (deprecated)');
    const linkedRun = wrapper.find('[data-testid="copied-from-link"]').at(0);
    expect(linkedRun.html()).toContain(ModelRegistryRoutes.getModelVersionPageRoute('Model B', '2'));
  });
});
