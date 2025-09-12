import { openDropdownMenu } from '@databricks/design-system/test-utils/rtl';
import { MemoryRouter } from '../../../common/utils/RoutingUtils';
import { MockedReduxStoreProvider } from '../../../common/utils/TestUtils';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import Utils from '../../../common/utils/Utils';
import type { ReduxState } from '../../../redux-types';
import { RunViewHeaderRegisterModelButton } from './RunViewHeaderRegisterModelButton';
import { DesignSystemProvider, DesignSystemThemeProvider } from '@databricks/design-system';
import userEvent from '@testing-library/user-event';
import { createModelVersionApi, createRegisteredModelApi } from '../../../model-registry/actions';
import type { KeyValueEntity } from '../../../common/types';

jest.mock('../../../model-registry/actions', () => ({
  searchRegisteredModelsApi: jest.fn(() => ({ type: 'MOCKED_ACTION', payload: Promise.resolve() })),
  createRegisteredModelApi: jest.fn(() => ({ type: 'MOCKED_ACTION', payload: Promise.resolve() })),
  createModelVersionApi: jest.fn(() => ({ type: 'MOCKED_ACTION', payload: Promise.resolve() })),
  searchModelVersionsApi: jest.fn(() => ({ type: 'MOCKED_ACTION', payload: Promise.resolve() })),
  getWorkspaceModelRegistryDisabledSettingApi: jest.fn(() => ({ type: 'MOCKED_ACTION', payload: Promise.resolve() })),
}));
const runUuid = 'testRunUuid';
const experimentId = 'testExperimentId';

// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(30000); // Larger timeout for integration testing

describe('RunViewHeaderRegisterModelButton integration', () => {
  const mountComponent = ({
    entities = {},
    tags = {},
    artifactRootUri,
  }: {
    artifactRootUri?: string;
    tags?: Record<string, KeyValueEntity>;
    entities?: Partial<Pick<ReduxState['entities'], 'modelVersionsByRunUuid'>>;
  } = {}) => {
    renderWithIntl(
      <MemoryRouter>
        <DesignSystemProvider>
          <MockedReduxStoreProvider
            state={{
              entities: {
                modelVersionsByRunUuid: {},
                modelByName: { 'existing-model': { name: 'existing-model', version: '1' } },
                ...entities,
              },
            }}
          >
            <div data-testid="container">
              <RunViewHeaderRegisterModelButton
                runTags={tags}
                artifactRootUri={artifactRootUri}
                runUuid={runUuid}
                experimentId={experimentId}
                registeredModelVersionSummaries={[]}
              />
            </div>
          </MockedReduxStoreProvider>
        </DesignSystemProvider>
      </MemoryRouter>,
    );
  };

  test('should render button and dropdown for multiple models, at least one unregistered and attempt to register a model', async () => {
    mountComponent({
      artifactRootUri: 'file://some/artifact/path',
      entities: {
        modelVersionsByRunUuid: {
          [runUuid]: [
            {
              source: `file://some/artifact/path/artifact_path`,
              version: '7',
              name: 'test-model',
            },
          ] as any,
        },
      },
      tags: {
        [Utils.loggedModelsTag]: {
          key: Utils.loggedModelsTag,
          value: JSON.stringify([
            {
              artifact_path: 'artifact_path',
              signature: {
                inputs: '[]',
                outputs: '[]',
                params: null,
              },
              flavors: {},
              run_id: runUuid,
              model_uuid: 12345,
            },
            {
              artifact_path: 'another_artifact_path',
              signature: {
                inputs: '[]',
                outputs: '[]',
                params: null,
              },
              flavors: {},
              run_id: runUuid,
              model_uuid: 12345,
            },
          ]),
        },
      },
    });

    await userEvent.type(screen.getByRole('button', { name: 'Register model' }), '{arrowdown}');

    await userEvent.click(screen.getByRole('menuitem', { name: /^another_artifact_path/ }));
    await userEvent.click(screen.getByText('Select a model'));

    await userEvent.click(screen.getByText('Create New Model'));

    await userEvent.click(screen.getByPlaceholderText('Input a model name'));
    await userEvent.paste('a-new-model');
    await userEvent.click(screen.getByRole('button', { name: 'Register' }));

    expect(createRegisteredModelApi).toHaveBeenCalledWith('a-new-model', expect.anything());
    expect(createModelVersionApi).toHaveBeenCalledWith(
      'a-new-model',
      'file://some/artifact/path/another_artifact_path',
      'testRunUuid',
      [],
      expect.anything(),
      undefined,
    );
  });
});
