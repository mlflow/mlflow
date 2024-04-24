import { openDropdownMenu } from '@databricks/design-system/test-utils/rtl';
import { MemoryRouter } from '../../../common/utils/RoutingUtils';
import { MockedReduxStoreProvider } from '../../../common/utils/TestUtils';
import { renderWithIntl, act, screen } from 'common/utils/TestUtils.react17';
import Utils from '../../../common/utils/Utils';
import { ReduxState } from '../../../redux-types';
import { RunViewHeaderRegisterModelButton } from './RunViewHeaderRegisterModelButton';
import { DesignSystemProvider, DesignSystemThemeProvider } from '@databricks/design-system';

jest.mock('../../../model-registry/actions', () => ({
  searchRegisteredModelsApi: jest.fn(() => ({ type: 'MOCKED_ACTION', payload: Promise.resolve() })),
  getWorkspaceModelRegistryDisabledSettingApi: jest.fn(() => ({ type: 'MOCKED_ACTION', payload: Promise.resolve() })),
}));

const runUuid = 'testRunUuid';
const experimentId = 'testExperimentId';

const testArtifactRootUriByRunUuid = { [runUuid]: 'file://some/artifact/path' };

const createModelArtifact = (artifactPath = 'random_forest_model') => ({
  artifact_path: artifactPath,
  signature: {
    inputs: '[]',
    outputs: '[]',
    params: null,
  },
  flavors: {},
  run_id: runUuid,
  model_uuid: 12345,
});

const createLoggedModelHistoryTag = (models: ReturnType<typeof createModelArtifact>[]) =>
  ({
    key: Utils.loggedModelsTag,
    value: JSON.stringify(models),
  } as any);

describe('RunViewHeaderRegisterModelButton', () => {
  const mountComponent = (
    entities: Partial<
      Pick<ReduxState['entities'], 'modelVersionsByRunUuid' | 'tagsByRunUuid' | 'artifactRootUriByRunUuid'>
    > = {},
  ) => {
    renderWithIntl(
      <MemoryRouter>
        <DesignSystemProvider>
          <MockedReduxStoreProvider
            state={{
              entities: {
                modelVersionsByRunUuid: {},
                tagsByRunUuid: {},
                artifactRootUriByRunUuid: testArtifactRootUriByRunUuid,
                ...entities,
              },
            }}
          >
            <div data-testid="container">
              <RunViewHeaderRegisterModelButton runUuid={runUuid} experimentId={experimentId} />
            </div>
          </MockedReduxStoreProvider>
        </DesignSystemProvider>
      </MemoryRouter>,
    );
  };
  test('should render nothing when there are no logged models', () => {
    mountComponent();
    expect(screen.getByTestId('container')).toBeEmptyDOMElement();
  });

  test('should render button for a single unregistered logged model', () => {
    mountComponent({
      tagsByRunUuid: {
        [runUuid]: {
          [Utils.loggedModelsTag]: createLoggedModelHistoryTag([createModelArtifact()]),
        },
      },
    });
    expect(screen.getByRole('button', { name: 'Register model' })).toBeInTheDocument();
  });

  test('should render simple link for a single registered logged model', () => {
    mountComponent({
      modelVersionsByRunUuid: {
        [runUuid]: [
          {
            source: `${testArtifactRootUriByRunUuid[runUuid]}/artifact_path`,
            version: '7',
            name: 'test-model',
          },
        ] as any,
      },
      tagsByRunUuid: {
        [runUuid]: {
          [Utils.loggedModelsTag]: createLoggedModelHistoryTag([createModelArtifact('artifact_path')]),
        },
      },
    });
    expect(screen.queryByRole('button', { name: 'Register model' })).not.toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Model registered' })).toHaveAttribute(
      'href',
      '/models/test-model/versions/7',
    );
  });

  test('should render button and dropdown for multiple models, all unregistered', async () => {
    mountComponent({
      tagsByRunUuid: {
        [runUuid]: {
          [Utils.loggedModelsTag]: createLoggedModelHistoryTag([
            createModelArtifact('artifact_path'),
            createModelArtifact('another_artifact_path'),
          ]),
        },
      },
    });
    expect(screen.getByRole('button', { name: 'Register model' })).toBeInTheDocument();

    await act(async () => {
      await openDropdownMenu(screen.getByRole('button', { name: 'Register model' }));
    });

    expect(screen.getByText('Unregistered models')).toBeInTheDocument();
    expect(screen.queryByText('Registered models')).not.toBeInTheDocument();
    expect(screen.getByRole('menuitem', { name: /^artifact_path/ })).toBeInTheDocument();
    expect(screen.getByRole('menuitem', { name: /^another_artifact_path/ })).toBeInTheDocument();
  });

  test('should render button and dropdown for multiple models, at least one unregistered', async () => {
    mountComponent({
      modelVersionsByRunUuid: {
        [runUuid]: [
          {
            source: `${testArtifactRootUriByRunUuid[runUuid]}/artifact_path`,
            version: '7',
            name: 'test-model',
          },
        ] as any,
      },
      tagsByRunUuid: {
        [runUuid]: {
          [Utils.loggedModelsTag]: createLoggedModelHistoryTag([
            createModelArtifact('artifact_path'),
            createModelArtifact('another_artifact_path'),
          ]),
        },
      },
    });

    expect(screen.getByRole('button', { name: 'Register model' })).toBeInTheDocument();

    await act(async () => {
      await openDropdownMenu(screen.getByRole('button', { name: 'Register model' }));
    });

    expect(screen.getByText('Unregistered models')).toBeInTheDocument();
    expect(screen.getByText('Registered models')).toBeInTheDocument();
    expect(screen.getByRole('menuitem', { name: /^another_artifact_path/ })).toBeInTheDocument();
    expect(screen.getByRole('menuitem', { name: /^test-model v7/ })).toBeInTheDocument();
  });

  test('should render button and dropdown for multiple models, all already registered', async () => {
    mountComponent({
      modelVersionsByRunUuid: {
        [runUuid]: [
          {
            source: `${testArtifactRootUriByRunUuid[runUuid]}/artifact_path`,
            version: '7',
            name: 'test-model',
          },
          {
            source: `${testArtifactRootUriByRunUuid[runUuid]}/another_artifact_path`,
            version: '8',
            name: 'another-test-model',
          },
        ] as any,
      },
      tagsByRunUuid: {
        [runUuid]: {
          [Utils.loggedModelsTag]: createLoggedModelHistoryTag([
            createModelArtifact('artifact_path'),
            createModelArtifact('another_artifact_path'),
          ]),
        },
      },
    });

    expect(screen.getByRole('button', { name: 'Register model' })).toBeInTheDocument();
    await act(async () => {
      await openDropdownMenu(screen.getByRole('button', { name: 'Register model' }));
    });

    expect(screen.queryByText('Unregistered models')).not.toBeInTheDocument();
    expect(screen.getByText('Registered models')).toBeInTheDocument();
    expect(screen.getByRole('menuitem', { name: /^another-test-model v8/ })).toBeInTheDocument();
    expect(screen.getByRole('menuitem', { name: /^test-model v7/ })).toBeInTheDocument();
  });
});
