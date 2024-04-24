import { openDropdownMenu } from '@databricks/design-system/test-utils/rtl';
import { MemoryRouter } from '../../../common/utils/RoutingUtils';
import { MockedReduxStoreProvider } from '../../../common/utils/TestUtils';
import { renderWithIntl, act, screen } from 'common/utils/TestUtils.react17';
import Utils from '../../../common/utils/Utils';
import { ReduxState } from '../../../redux-types';
import { RunViewHeaderRegisterModelButton } from './RunViewHeaderRegisterModelButton';
import { DesignSystemProvider, DesignSystemThemeProvider } from '@databricks/design-system';
import userEvent from '@testing-library/user-event';
import { createModelVersionApi, createRegisteredModelApi } from '../../../model-registry/actions';

jest.mock('../../../model-registry/actions', () => ({
  searchRegisteredModelsApi: jest.fn(() => ({ type: 'MOCKED_ACTION', payload: Promise.resolve() })),
  createRegisteredModelApi: jest.fn(() => ({ type: 'MOCKED_ACTION', payload: Promise.resolve() })),
  createModelVersionApi: jest.fn(() => ({ type: 'MOCKED_ACTION', payload: Promise.resolve() })),
  searchModelVersionsApi: jest.fn(() => ({ type: 'MOCKED_ACTION', payload: Promise.resolve() })),
  getWorkspaceModelRegistryDisabledSettingApi: jest.fn(() => ({ type: 'MOCKED_ACTION', payload: Promise.resolve() })),
}));
const runUuid = 'testRunUuid';
const experimentId = 'testExperimentId';

const testArtifactRootUriByRunUuid = { [runUuid]: 'file://some/artifact/path' };

jest.setTimeout(30000); // Larget timeout for integration testing

describe('RunViewHeaderRegisterModelButton integration', () => {
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
                modelByName: { 'existing-model': { name: 'existing-model', version: '1' } },
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

  test('should render button and dropdown for multiple models, at least one unregistered and attempt to register a model', async () => {
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
          } as any,
        },
      },
    });

    await act(async () => {
      await openDropdownMenu(screen.getByRole('button', { name: 'Register model' }));
    });

    await act(async () => {
      userEvent.click(screen.getByRole('menuitem', { name: /^another_artifact_path/ }));
      userEvent.click(screen.getByText('Select a model'));
    });

    await act(async () => {
      userEvent.click(screen.getByText('Create New Model'));
    });

    await act(async () => {
      userEvent.paste(screen.getByPlaceholderText('Input a model name'), 'a-new-model');
      userEvent.click(screen.getByRole('button', { name: 'Register' }));
    });

    expect(createRegisteredModelApi).toBeCalledWith('a-new-model', expect.anything());
    expect(createModelVersionApi).toBeCalledWith(
      'a-new-model',
      'file://some/artifact/path/another_artifact_path',
      'testRunUuid',
      [],
      expect.anything(),
    );
  });
});
