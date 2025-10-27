import { render, screen } from '@testing-library/react';
import { ExperimentViewRunsControlsActionsSelectTags } from './ExperimentViewRunsControlsActionsSelectTags';
import type { RunInfoEntity } from '../../../../types';
import type { KeyValueEntity } from '../../../../../common/types';
import { IntlProvider } from 'react-intl';
import { MemoryRouter } from '../../../../../common/utils/RoutingUtils';
import { setRunTagsBulkApi } from '@mlflow/mlflow/src/experiment-tracking/actions';
import { MockedReduxStoreProvider } from '../../../../../common/utils/TestUtils';
import userEvent from '@testing-library/user-event';

jest.mock('@mlflow/mlflow/src/experiment-tracking/actions', () => ({
  setRunTagsBulkApi: jest.fn(() => ({ type: 'setRunTagsBulkApi', payload: Promise.resolve() })),
}));

// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(10000);

describe('ExperimentViewRunsControlsActionsSelectTags', () => {
  beforeEach(() => {
    jest.mocked(setRunTagsBulkApi).mockClear();
  });

  const runsSelected = {
    runUuid1: true,
    runUuid2: false,
    runUuid3: true,
  };

  const runInfos = [{ runUuid: 'runUuid1' }, { runUuid: 'runUuid2' }, { runUuid: 'runUuid3' }] as RunInfoEntity[];

  const tagsList = [
    {
      tag1: {
        key: 'tag1',
        value: 'test1',
      },
      tag2: {
        key: 'tag2',
        value: 'test2',
      },
      // Doesn't have tag3
    },
    {
      tag1: {
        key: 'tag1',
        value: 'test2',
      },
      tag3: {
        key: 'tag3',
        value: 'test3',
      },
    },
    {
      tag2: {
        key: 'tag2',
        value: 'test1',
      },
      tag3: {
        key: 'tag3',
        value: 'test3',
      },
    },
  ] as Record<string, KeyValueEntity>[];

  const refreshRuns = jest.fn();

  const renderComponent = () => {
    render(
      <MockedReduxStoreProvider>
        <MemoryRouter>
          <IntlProvider locale="en">
            <ExperimentViewRunsControlsActionsSelectTags
              runInfos={runInfos}
              runsSelected={runsSelected}
              tagsList={tagsList}
              refreshRuns={refreshRuns}
            />
          </IntlProvider>
        </MemoryRouter>
      </MockedReduxStoreProvider>,
    );
  };

  it('renders the component correctly', () => {
    renderComponent();
    // Assert that the component renders without errors
    expect(screen.getByText('Add tags')).toBeInTheDocument();
  });

  it('opens the dropdown when the trigger is clicked', async () => {
    renderComponent();

    // Click on the trigger to open the dropdown
    await userEvent.click(screen.getByTestId('runs-tag-multiselect-trigger'));

    // Assert that the dropdown is open
    expect(screen.getByRole('listbox')).toBeInTheDocument();
  });

  it('selects and deselects tags correctly', async () => {
    renderComponent();

    // Click on the trigger to open the dropdown
    await userEvent.click(screen.getByTestId('runs-tag-multiselect-trigger'));

    // expect(screen.getByText("tag1: test1")).toBePartiallyChecked();
    await userEvent.click(screen.getByRole('option', { name: 'tag1: test1' }));
    expect(screen.getByRole('checkbox', { name: 'tag1: test1' })).toBeChecked();
    await userEvent.click(screen.getByRole('option', { name: 'tag1: test1' }));
    expect(screen.getByRole('checkbox', { name: 'tag1: test1' })).not.toBeChecked();
  });

  it("adds a new tag when the 'Add new tag' button is clicked", async () => {
    renderComponent();

    // Click on the trigger to open the dropdown
    await userEvent.click(screen.getByTestId('runs-tag-multiselect-trigger'));

    // Click on the 'Add new tag' button
    await userEvent.click(screen.getByTestId('runs-add-new-tag-button'));

    // Assert that the add new tag modal is open
    expect(screen.getByLabelText('Add New Tag')).toBeInTheDocument();

    // Input a new tag key and value into the modal input boxes
    await userEvent.type(screen.getByTestId('add-new-tag-key-input'), 'newTag');
    await userEvent.type(screen.getByTestId('add-new-tag-value-input'), 'newValue');

    // Click on the 'Add' button
    await userEvent.click(screen.getByText('Add'));

    // Assert that it shows up on the dropdown
    expect(screen.getByRole('option', { name: 'newTag: newValue' })).toBeInTheDocument();
  });

  it("saves the selected tags when the 'Save' button is clicked", async () => {
    renderComponent();

    // Click on the trigger to open the dropdown
    await userEvent.click(screen.getByTestId('runs-tag-multiselect-trigger'));

    // Select some tags
    await userEvent.click(screen.getByRole('option', { name: 'tag1: test1' }));
    await userEvent.click(screen.getByRole('option', { name: 'tag2: test2' }));
    await userEvent.click(screen.getByRole('option', { name: 'tag2: test1' }));
    await userEvent.click(screen.getByRole('option', { name: 'tag2: test1' }));

    // Click on the 'Save' button
    await userEvent.click(screen.getByText('Save'));

    // Two runs are selected
    expect(setRunTagsBulkApi).toHaveBeenCalledTimes(2);
    // Assert the function was called with the correct arguments
    expect(setRunTagsBulkApi).toHaveBeenCalledWith(
      'runUuid1',
      [
        { key: 'tag1', value: 'test1' },
        { key: 'tag2', value: 'test2' },
      ],
      [
        { key: 'tag1', value: 'test1' },
        { key: 'tag2', value: 'test2' },
      ],
    );

    expect(setRunTagsBulkApi).toHaveBeenCalledWith(
      'runUuid3',
      [
        { key: 'tag2', value: 'test1' },
        { key: 'tag3', value: 'test3' },
      ],
      [
        { key: 'tag1', value: 'test1' },
        { key: 'tag2', value: 'test2' },
        { key: 'tag3', value: 'test3' },
      ],
    );
  });
});
