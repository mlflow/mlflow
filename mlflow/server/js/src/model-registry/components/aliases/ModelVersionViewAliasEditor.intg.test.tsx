import userEvent from '@testing-library/user-event';

import { ModelVersionViewAliasEditor } from './ModelVersionViewAliasEditor';
import { renderWithIntl, act, screen, within, findAntdOption } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { Provider } from 'react-redux';

import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';

import { useEffect, useState } from 'react';
import type { ModelEntity, ModelVersionInfoEntity } from '../../../experiment-tracking/types';
import { Services as ModelRegistryServices } from '../../services';

// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(60000); // increase timeout since it's integration testing

/**
 * A simple mocked API to that mimics storing registered model,
 * its versions and and is capable of updating aliases
 */
class MockDatabase {
  models: ModelEntity[] = [];
  modelVersions: ModelVersionInfoEntity[] = [];
  constructor() {
    const initialAliases = { 1: ['champion', 'first_version'], 2: ['challenger'] };
    const initialModel: ModelEntity = {
      name: 'test_model',
      creation_timestamp: 1234,
      last_updated_timestamp: 2345,
      current_stage: '',
      email_subscription_status: 'active',
      permission_level: '',
      source: 'notebook',
      status: 'active',
      version: '1',
      description: '',
      id: 'test_model',
      latest_versions: [],
      tags: [],
    };
    const initialModelVersions: ModelVersionInfoEntity[] = [
      {
        ...initialModel,
        last_updated_timestamp: 1234,
        run_id: 'experiment123456789_run4',
        user_id: '123',
        version: '1',
        tags: [],
        aliases: initialAliases['1'],
      },
      {
        ...initialModel,
        last_updated_timestamp: 1234,
        run_id: 'experiment123456789_run4',
        user_id: '123',
        version: '2',
        tags: [],
        aliases: initialAliases['2'],
      },
    ];
    initialModel.latest_versions.push(initialModelVersions[1]);

    this.models.push(initialModel);
    this.modelVersions.push(...initialModelVersions);

    initialModel.aliases = this.modelVersions.flatMap(({ version, aliases }) =>
      (aliases || []).map((alias) => ({ version, alias })),
    );
  }
  private getModel(findName: string) {
    return this.models.find(({ name }) => name === findName) || null;
  }
  private getVersions(modelName: string) {
    return this.modelVersions.filter(({ name, version }) => name === modelName) || [];
  }
  // Exposed "API":
  fetchModel = async (findName: string) => {
    return Promise.resolve(this.getModel(findName));
  };
  searchModelVersions = async (modelName: string) => {
    return Promise.resolve(this.getVersions(modelName));
  };
  setAlias = async ({ alias, name, version }: { alias: string; name: string; version: string }) => {
    const existingAliasVersion = this.modelVersions.find(
      (versionEntity) => versionEntity.name === name && versionEntity.aliases?.includes(alias),
    );
    if (existingAliasVersion) {
      existingAliasVersion.aliases = existingAliasVersion.aliases?.filter((a) => a !== alias) || [];
    }
    const aliasVersion = this.modelVersions.find(
      (versionEntity) => versionEntity.name === name && versionEntity.version === version,
    );
    if (!aliasVersion) {
      return;
    }
    aliasVersion.aliases = Array.from(new Set([...(aliasVersion.aliases || []), alias]));
    const model = this.models.find((existingModel) => existingModel.name === name);
    if (model) {
      model.aliases = this.modelVersions.flatMap(({ version: existingVersion, aliases: existingAliases }) =>
        (existingAliases || []).map((a) => ({ version: existingVersion, alias: a })),
      );
    }
  };
  deleteAlias = async ({ alias, name, version }: { alias: string; name: string; version: string }) => {
    const existingAliasVersion = this.modelVersions.find(
      (versionEntity) => versionEntity.name === name && versionEntity.version === version,
    );
    if (existingAliasVersion) {
      existingAliasVersion.aliases = existingAliasVersion.aliases?.filter((a) => a !== alias) || [];
    }

    const model = this.models.find((existingModel) => existingModel.name === name);
    if (model) {
      model.aliases = this.modelVersions.flatMap(({ version: existingVersion, aliases: existingAliases }) =>
        (existingAliases || []).map((a) => ({ version: existingVersion, alias: a })),
      );
    }
  };
}

/**
 * This integration test tests component, hook and service
 * logic for adding and deleting model aliases.
 *
 * Test scenario:
 * - Load model "test_model" from mocked API
 * - Load two model versions:
 *   - Version "1" with added 'champion', 'first_version' aliases
 *   - Version "2" with added 'challenger' alias
 * - View version #2 aliases
 * - Open alias editor modal for version #2
 * - Add "champion" alias
 * - Add "latest_version" alias
 * - Save the data
 * - Confirm that new aliases are added and old ones are gone
 * - View version #1 aliases
 * - Confirm that aliases have been reloaded
 * - Open alias editor modal for version #1
 * - Remove "first_version" alias
 * - Save the data
 * - Confirm that aliases are removed
 * - Given no aliases exist for version #1, confirm the button with title "Add aliases" exists
 */
describe('useEditRegisteredModelAliasesModal integration', () => {
  // Wire up service to the mocked "database" server
  const database = new MockDatabase();
  ModelRegistryServices.setModelVersionAlias = jest.fn(database.setAlias);
  ModelRegistryServices.deleteModelVersionAlias = jest.fn(database.deleteAlias);

  let unmountPage: () => void;

  function renderTestPage(currentVersionNumber: string) {
    // Mock redux store to enable redux actions
    const mockStoreFactory = configureStore([thunk, promiseMiddleware()]);
    const mockStore = mockStoreFactory({});

    function TestPage() {
      const [model, setModel] = useState<ModelEntity | null>(null);
      const [versions, setVersions] = useState<ModelVersionInfoEntity[]>([]);

      const fetchData = () => {
        database.fetchModel('test_model').then(setModel);
        database.searchModelVersions('test_model').then(setVersions);
      };

      useEffect(() => {
        fetchData();
      }, []);

      const editedVersion = versions.find(({ version }) => version === currentVersionNumber);

      if (!model || !editedVersion) {
        return <>Loading</>;
      }

      return (
        <ModelVersionViewAliasEditor
          version={currentVersionNumber}
          modelEntity={model}
          aliases={editedVersion.aliases}
          onAliasesModified={fetchData}
        />
      );
    }
    const { unmount } = renderWithIntl(
      <Provider store={mockStore}>
        <TestPage />
      </Provider>,
    );
    unmountPage = unmount;
  }

  test('it should display model details and modify its aliases using the modal', async () => {
    // Render the page for model version #2
    await act(async () => {
      renderTestPage('2');
    });

    // Check if the "challenger" alias exists
    expect(screen.getByRole('status', { name: 'challenger' })).toBeInTheDocument();

    // Open the editor modal
    await userEvent.click(screen.getByRole('button', { name: 'Edit aliases' }));

    // Type in "champion" alias name, confirm the selection
    await userEvent.type(screen.getByRole('combobox'), 'champion');
    await userEvent.click(await findAntdOption('champion'));

    // Assert there's a conflict
    expect(
      screen.getByText(
        'The "champion" alias is also being used on version 1. Adding it to this version will remove it from version 1.',
      ),
    ).toBeInTheDocument();

    // Add yet another alias, confirm the selection
    await userEvent.type(screen.getByRole('combobox'), 'latest_version');
    await userEvent.click(await findAntdOption('latest_version'));

    // Save the aliases
    await userEvent.click(screen.getByRole('button', { name: 'Save aliases' }));

    // Assert there are new aliases loaded from "API"
    expect(screen.getByRole('status', { name: 'champion' })).toBeInTheDocument();
    expect(screen.getByRole('status', { name: 'challenger' })).toBeInTheDocument();
    expect(screen.getByRole('status', { name: 'latest_version' })).toBeInTheDocument();

    // Unmount the page, display version #1 aliases now
    await act(async () => {
      unmountPage();
      renderTestPage('1');
    });

    // Assert there are version 1 aliases shown only
    expect(screen.getByRole('status', { name: 'first_version' })).toBeInTheDocument();
    expect(screen.queryByRole('status', { name: 'champion' })).not.toBeInTheDocument();
    expect(screen.queryByRole('status', { name: 'latest_version' })).not.toBeInTheDocument();

    // Show the editor modal
    await userEvent.click(screen.getByRole('button', { name: 'Edit aliases' }));

    // Locate the tag pill with the existing "first_version" alias, click the "X" button within
    await userEvent.click(
      within(within(screen.getByRole('dialog')).getByRole('status', { name: 'first_version' })).getByRole('button'),
    );

    // Save the new aliases
    await userEvent.click(screen.getByRole('button', { name: 'Save aliases' }));

    // Confirm there are no aliases shown at all
    expect(screen.queryByRole('status')).not.toBeInTheDocument();

    // Confirm a button with "Add aliases" title is displayed now
    expect(screen.queryByTitle('Add aliases')).toBeInTheDocument();
  });
});
