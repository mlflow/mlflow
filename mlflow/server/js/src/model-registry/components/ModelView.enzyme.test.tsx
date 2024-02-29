/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { ModelView, ModelViewImpl, StageFilters } from './ModelView';
import { mockModelVersionDetailed, mockRegisteredModelDetailed } from '../test-utils';
import { ModelVersionStatus, Stages } from '../constants';
import { MemoryRouter } from '../../common/utils/RoutingUtils';
import { ModelVersionTable } from './ModelVersionTable';
import Utils from '../../common/utils/Utils';
import { ModelRegistryRoutes } from '../routes';
import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import { RegisteredModelTag } from '../sdk/ModelRegistryMessages';
import { Provider } from 'react-redux';
import { mountWithIntl } from 'common/utils/TestUtils.enzyme';
import { renderWithIntl, screen } from 'common/utils/TestUtils.react17';
import { DesignSystemProvider } from '@databricks/design-system';
describe('ModelView', () => {
  let wrapper: any;
  let instance;
  let minimalProps: any;
  let minimalStoreRaw;
  let minimalStore: any;
  let createComponentInstance: any;
  const navigateMock = jest.fn();
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
        'special key': (RegisteredModelTag as any).fromJs({
          key: 'special key',
          value: 'not so special value',
        }),
      },
    ],
  };
  beforeEach(() => {
    // TODO: remove global fetch mock by explicitly mocking all the service API calls
    // @ts-expect-error TS(2322): Type 'Mock<Promise<{ ok: true; status: number; tex... Remove this comment to see the full error message
    global.fetch = jest.fn(() => Promise.resolve({ ok: true, status: 200, text: () => Promise.resolve('') }));
    navigateMock.mockClear();
    minimalProps = {
      model: mockRegisteredModelDetailed(
        mockModel.name,
        // @ts-expect-error TS(2345): Argument of type '{ name: any; creation_timestamp:... Remove this comment to see the full error message
        mockModel.latestVersions,
        mockModel.tags,
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // @ts-ignore - OSS specific ignore
        mockModel.permissionLevel,
      ),
      modelVersions: mockModel.versions,
      handleEditDescription: jest.fn(),
      handleDelete: jest.fn(),
      showEditPermissionModal: jest.fn(),
      navigate: navigateMock,
      tags: {},
      setRegisteredModelTagApi: jest.fn(),
      deleteRegisteredModelTagApi: jest.fn(),
    };
    minimalStoreRaw = {
      entities: {
        tagsByRegisteredModel: {
          'Model A': {
            'special key': (RegisteredModelTag as any).fromJs({
              key: 'special key',
              value: 'not so special value',
            }),
          },
        },
      },
      apis: {},
    };
    minimalStore = mockStore(minimalStoreRaw);
    createComponentInstance = (modelViewProps: any) =>
      mountWithIntl(
        <DesignSystemProvider>
          <Provider store={minimalStore}>
            <MemoryRouter>
              <ModelView {...modelViewProps} />
            </MemoryRouter>
          </Provider>
        </DesignSystemProvider>,
      );
    // This hides the useNextModelsUI promotion modal
    jest.spyOn(window.localStorage, 'getItem').mockImplementation((key) => (key.match(/promo/) ? 'true' : ''));
  });
  const mockModelsNextUIDismissed = () =>
    jest
      .spyOn(window.localStorage, 'getItem')
      .mockImplementation((key) => (key.match(/dismiss_next_model_registry/) ? 'true' : ''));
  test('should render with minimal props without exploding', () => {
    wrapper = createComponentInstance(minimalProps);
    expect(wrapper.find(ModelView).length).toBe(1);
  });
  test('should render all model versions initially', () => {
    wrapper = createComponentInstance(minimalProps);
    expect(wrapper.find('div[role="cell"].model-version').length).toBe(3);
    expect(wrapper.find('div[role="cell"].model-version').at(0).text()).toBe('Version 3');
    expect(wrapper.find('div[role="cell"].model-version').at(1).text()).toBe('Version 2');
    expect(wrapper.find('div[role="cell"].model-version').at(2).text()).toBe('Version 1');
  });
  test('should render model version table with activeStageOnly when "Active" button is on', async () => {
    mockModelsNextUIDismissed();
    wrapper = createComponentInstance(minimalProps);
    instance = wrapper.find(ModelViewImpl).instance();
    expect(wrapper.find(ModelVersionTable).props().activeStageOnly).toBe(false);
    instance.setState({ stageFilter: StageFilters.ACTIVE });
    wrapper.update();
    expect(wrapper.find(ModelVersionTable).props().activeStageOnly).toBe(true);
  });
  test('Page title is set', () => {
    const mockUpdatePageTitle = jest.fn();
    Utils.updatePageTitle = mockUpdatePageTitle;
    wrapper = createComponentInstance(minimalProps);
    expect(mockUpdatePageTitle.mock.calls[0][0]).toBe('Model A - MLflow Model');
  });
  test('should disable dropdown delete menu item when model has active versions', () => {
    const props = {
      ...minimalProps,
      model: {
        ...minimalProps.model,
      },
    };
    wrapper = createComponentInstance(props);
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
    wrapper = createComponentInstance(minimalProps);
    expect(wrapper.find('[data-test-id="compareButton"]').hostNodes().length).toBe(1);
    expect(wrapper.find('[data-test-id="compareButton"]').hostNodes().props().disabled).toEqual(true);
    wrapper
      .find(ModelViewImpl)
      .instance()
      .setState({
        runsSelected: { run_id_1: 'version_1' },
      });
    wrapper.update();
    expect(wrapper.find('[data-test-id="compareButton"]').hostNodes().props().disabled).toEqual(true);
    const twoRunsSelected = { run_id_1: 'version_1', run_id_2: 'version_2' };
    wrapper.find(ModelViewImpl).instance().setState({
      runsSelected: twoRunsSelected,
    });
    wrapper.update();
    expect(wrapper.find('[data-test-id="compareButton"]').hostNodes().props().disabled).toEqual(false);
    wrapper.find('[data-test-id="compareButton"]').hostNodes().simulate('click');
    expect(navigateMock).toHaveBeenCalledWith(
      ModelRegistryRoutes.getCompareModelVersionsPageRoute(minimalProps['model']['name'], twoRunsSelected),
    );
  });
  test('should tags rendered in the UI', () => {
    wrapper = createComponentInstance(minimalProps);
    expect(wrapper.html()).toContain('special key');
    expect(wrapper.html()).toContain('not so special value');
  });
  test('creator description not rendered if user_id is unavailable', () => {
    wrapper = createComponentInstance(minimalProps);
    expect(wrapper.find('[data-testid="model-view-metadata-item"]').length).toBe(2);
    expect(wrapper.find('[data-testid="model-view-metadata"]').text()).toContain('Created Time');
    expect(wrapper.find('[data-testid="model-view-metadata"]').text()).toContain('Last Modified');
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
    wrapper = createComponentInstance(props);
    expect(wrapper.find('[data-testid="model-view-metadata-item"]').length).toBe(3);
    expect(wrapper.find('[data-testid="model-view-metadata"]').text()).toContain('Creator');
    expect(wrapper.find('[data-testid="model-view-metadata"]').text()).toContain(user_id);
  });
  test('should display aliases and tags columns instead of stage in new models UI', async () => {
    renderWithIntl(
      <DesignSystemProvider>
        <Provider store={minimalStore}>
          <MemoryRouter>
            <ModelView {...minimalProps} />
          </MemoryRouter>
        </Provider>
      </DesignSystemProvider>,
    );

    expect(screen.queryByRole('columnheader', { name: 'Stage' })).not.toBeInTheDocument();
    expect(screen.queryByRole('radio', { name: /Active/ })).not.toBeInTheDocument();
    expect(screen.queryByRole('columnheader', { name: 'Aliases' })).toBeInTheDocument();
    expect(screen.queryByRole('columnheader', { name: 'Tags' })).toBeInTheDocument();
  });

  test('should display stage column instead of aliases and tags when old models UI is used', async () => {
    mockModelsNextUIDismissed();

    renderWithIntl(
      <DesignSystemProvider>
        <Provider store={minimalStore}>
          <MemoryRouter>
            <ModelView {...minimalProps} />
          </MemoryRouter>
        </Provider>
      </DesignSystemProvider>,
    );

    expect(screen.queryByRole('columnheader', { name: 'Stage' })).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: /Active \d*/ })).toBeInTheDocument();
    expect(screen.queryByRole('columnheader', { name: 'Aliases' })).not.toBeInTheDocument();
    expect(screen.queryByRole('columnheader', { name: 'Tags' })).not.toBeInTheDocument();
  });
});
