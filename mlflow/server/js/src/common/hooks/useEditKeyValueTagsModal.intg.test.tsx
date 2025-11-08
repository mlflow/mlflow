import userEvent from '@testing-library/user-event';

import { useEffect, useState } from 'react';
import { Provider, useDispatch } from 'react-redux';
import configureStore from 'redux-mock-store';
import promiseMiddleware from 'redux-promise-middleware';
import thunk from 'redux-thunk';
import type { ModelVersionInfoEntity } from '../../experiment-tracking/types';
import { updateModelVersionTagsApi } from '../../model-registry/actions';
import { Services as ModelRegistryServices } from '../../model-registry/services';
import type { ThunkDispatch } from '../../redux-types';
import { act, screen, within, fastFillInput, renderWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { useEditKeyValueTagsModal } from './useEditKeyValueTagsModal';

const ERRONEOUS_TAG_KEY = 'forbidden_tag';

// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(90000); // increase timeout since it's integration testing

/**
 * A super simple mocked API to that mimics storing registered model
 * versions and and is capable of updating the tags
 */
class MockDatabase {
  // Internal storage:
  modelVersions: ModelVersionInfoEntity[] = [
    {
      name: 'test_model',
      creation_timestamp: 1234,
      current_stage: '',
      last_updated_timestamp: 1234,
      run_id: 'experiment123456789_run4',
      source: 'notebook',
      status: 'active',
      user_id: '123',
      version: '1',
      tags: [
        { key: 'existing_tag_1', value: 'existing_tag_value_1' },
        { key: 'existing_tag_2', value: 'existing_tag_value_2' },
      ] as any,
    },
  ];
  private getVersion(findName: string, findVersion: string) {
    return this.modelVersions.find(({ name, version }) => name === findName && version === findVersion) || null;
  }
  // Exposed "API":
  getModelVersion = async (findName: string, findVersion: string) => {
    return Promise.resolve(this.getVersion(findName, findVersion));
  };
  setTag = async ({ name, key, version, value }: { name: string; version: string; key: string; value?: string }) => {
    if (key === ERRONEOUS_TAG_KEY) {
      throw new Error('You shall not use this tag!');
    }
    const modelVersion = this.getVersion(name, version);
    if (!modelVersion) {
      return;
    }
    const existingTag = modelVersion.tags?.find(({ key: existingKey }) => existingKey === key);
    if (existingTag && value) {
      existingTag.value = value;
      return;
    }
    modelVersion.tags?.push({ key, value: value || '' } as any);
  };
  deleteTag = async ({ name, key, version }: { name: string; version: string; key: string }) => {
    const modelVersion = this.getVersion(name, version);
    if (!modelVersion) {
      return;
    }
    modelVersion.tags = (modelVersion.tags || []).filter(({ key: existingKey }) => existingKey !== key);
  };
}

describe('useEditKeyValueTagsModal integration', () => {
  // Wire up service to the mocked "database" server
  const database = new MockDatabase();
  ModelRegistryServices.deleteModelVersionTag = jest.fn(database.deleteTag);
  ModelRegistryServices.setModelVersionTag = jest.fn(database.setTag);

  // Mock redux store to enable redux actions
  const mockStoreFactory = configureStore([thunk, promiseMiddleware()]);
  const mockStore = mockStoreFactory({});

  function renderTestComponent() {
    function TestComponent() {
      const dispatch = useDispatch<ThunkDispatch>();
      const [modelVersion, setModelVersion] = useState<ModelVersionInfoEntity | null>(null);

      const fetchModelVersion = () => database.getModelVersion('test_model', '1').then(setModelVersion);
      useEffect(() => {
        fetchModelVersion();
      }, []);

      const { showEditTagsModal, EditTagsModal } = useEditKeyValueTagsModal<ModelVersionInfoEntity>({
        allAvailableTags: [],
        saveTagsHandler: async (savedModelVersion, existingTags, newTags) =>
          dispatch(updateModelVersionTagsApi(savedModelVersion, existingTags, newTags)),
        onSuccess: fetchModelVersion,
      });
      return (
        <>
          {modelVersion && (
            <div>
              <span>
                Model name: {modelVersion.name} version {modelVersion.version}
              </span>
              <ul>
                {modelVersion.tags?.map(({ key, value }) => (
                  <li key={key} title={key}>
                    {key}
                    {value && `: ${value}`}
                  </li>
                ))}
              </ul>
            </div>
          )}
          <button onClick={() => modelVersion && showEditTagsModal(modelVersion)}>change tags</button>
          {EditTagsModal}
        </>
      );
    }
    renderWithIntl(
      <Provider store={mockStore}>
        <TestComponent />
      </Provider>,
    );
  }

  test('it should display model details and modify its tags using the modal', async () => {
    // Render the component, wait to load initial data
    await act(async () => {
      renderTestComponent();
    });

    // Assert existence of existing tags
    expect(screen.getByText('Model name: test_model version 1')).toBeInTheDocument();
    expect(screen.getByRole('listitem', { name: 'existing_tag_1' })).toContainHTML(
      'existing_tag_1: existing_tag_value_1',
    );
    expect(screen.getByRole('listitem', { name: 'existing_tag_2' })).toContainHTML(
      'existing_tag_2: existing_tag_value_2',
    );

    // Open the modal in order to modify tags
    await userEvent.click(screen.getByRole('button', { name: 'change tags' }));
    expect(screen.getByRole('dialog', { name: /Add\/Edit tags/ })).toBeInTheDocument();

    // Add a new tag with value
    await fastFillInput(within(screen.getByRole('dialog')).getByRole('combobox'), 'new_tag_with_value');
    await userEvent.click(screen.getByText(/Add tag "new_tag_with_value"/));
    await fastFillInput(screen.getByLabelText('Value (optional)'), 'tag_value');
    await userEvent.click(screen.getByLabelText('Add tag'));

    // Add another tag without value
    await fastFillInput(within(screen.getByRole('dialog')).getByRole('combobox'), 'new_tag_without_value');

    await userEvent.click(screen.getByText(/Add tag "new_tag_without_value"/));
    await userEvent.click(screen.getByLabelText('Add tag'));

    // Add yet another tag without value
    await fastFillInput(within(screen.getByRole('dialog')).getByRole('combobox'), 'another_tag');
    await userEvent.click(screen.getByText(/Add tag "another_tag"/));
    await userEvent.click(screen.getByLabelText('Add tag'));

    // Delete existing tag
    await userEvent.click(within(screen.getByRole('status', { name: 'existing_tag_1' })).getByRole('button'));
    // Also, scratch one of the newly added tags
    await userEvent.click(within(screen.getByRole('status', { name: 'another_tag' })).getByRole('button'));

    // Save the tags
    await userEvent.click(screen.getByRole('button', { name: 'Save tags' }));

    // Assert tags from newly refreshed model version
    expect(screen.queryByRole('listitem', { name: 'existing_tag_1' })).not.toBeInTheDocument();
    expect(screen.queryByRole('listitem', { name: 'existing_tag_2' })).toBeInTheDocument();
    expect(screen.getByRole('listitem', { name: 'new_tag_without_value' })).toContainHTML('new_tag_without_value');
    expect(screen.getByRole('listitem', { name: 'new_tag_with_value' })).toContainHTML('new_tag_with_value: tag_value');
    expect(screen.queryByRole('listitem', { name: 'another_tag' })).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'change tags' }));
    expect(screen.getByRole('dialog', { name: /Add\/Edit tags/ })).toBeInTheDocument();
  });
  test('should react accordingly when API responds with an error', async () => {
    // Render the component, wait to load initial data
    await act(async () => {
      renderTestComponent();
    });

    // Open the modal in order to modify tags
    await userEvent.click(screen.getByRole('button', { name: 'change tags' }));

    await fastFillInput(within(screen.getByRole('dialog')).getByRole('combobox'), 'forbidden_tag');

    await userEvent.click(screen.getByText(/Add tag "forbidden_tag"/));
    await userEvent.click(screen.getByLabelText('Add tag'));

    // Attempt to save it
    await userEvent.click(screen.getByRole('button', { name: 'Save tags' }));

    // Confirm there's an error and that the tag was not added
    expect(screen.getByText(/You shall not use this tag!/)).toBeInTheDocument();
    expect(screen.queryByRole('listitem', { name: 'forbidden_tag' })).not.toBeInTheDocument();
  });
});
