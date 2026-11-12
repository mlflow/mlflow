import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { PointerEventsCheckLevel } from '@testing-library/user-event';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEventGlobal from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { useCheckMultiturnDatasets } from '../../experiment-evaluation-datasets/hooks/useCheckMultiturnDatasets';
import { useSearchEvaluationDatasets } from '../../experiment-evaluation-datasets/hooks/useSearchEvaluationDatasets';
import { useUpsertDatasetRecordsMutation } from '../../experiment-evaluation-datasets/hooks/useUpsertDatasetRecordsMutation';
import type { ConversationMessage } from '../types';
import { AddToDatasetModal } from './AddToDatasetModal';

jest.mock('../../experiment-evaluation-datasets/hooks/useSearchEvaluationDatasets', () => ({
  useSearchEvaluationDatasets: jest.fn(),
}));
jest.mock('../../experiment-evaluation-datasets/hooks/useCheckMultiturnDatasets', () => ({
  useCheckMultiturnDatasets: jest.fn(),
}));
jest.mock('../../experiment-evaluation-datasets/hooks/useUpsertDatasetRecordsMutation', () => ({
  useUpsertDatasetRecordsMutation: jest.fn(),
}));
// The create-dataset button opens its own modal + mutation; stub it with a button that
// reports a created dataset, to keep this test focused on what the picker does with it.
jest.mock('../../experiment-evaluation-datasets/components/CreateEvaluationDatasetButton', () => ({
  CreateEvaluationDatasetButton: ({
    onCreated,
  }: {
    onCreated?: (dataset: { dataset_id: string; name: string }) => void;
  }) => (
    <button type="button" onClick={() => onCreated?.({ dataset_id: 'd-new', name: 'New Dataset' })}>
      Create dataset
    </button>
  ),
}));

// Modal overlays mask elements behind pointer-events checks; disable so userEvent can click through.
const userEvent = userEventGlobal.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });

const mockSearchDatasets = jest.mocked(useSearchEvaluationDatasets);
const mockCheckMultiturn = jest.mocked(useCheckMultiturnDatasets);
const mockUpsertMutation = jest.mocked(useUpsertDatasetRecordsMutation);

const upsertDatasetRecordsMutationAsync = jest.fn<(payload: { datasetId: string; records: string }) => Promise<any>>();
const invalidateAfterUpsert = jest.fn();

const mockUpsert = (mutationAsync: typeof upsertDatasetRecordsMutationAsync) =>
  mockUpsertMutation.mockReturnValue({
    upsertDatasetRecordsMutationAsync: mutationAsync,
    invalidateAfterUpsert,
    isLoading: false,
  } as any);

const HUMAN_SOURCE = { source_type: 'HUMAN', source_data: {} };

const CONVERSATION: ConversationMessage[] = [
  { role: 'user', content: 'What is MLflow?' },
  {
    role: 'assistant',
    content: 'MLflow is an ML platform.',
    requestMessages: [{ role: 'user', content: 'What is MLflow?' }],
  },
  { role: 'user', content: '' },
];

type TestDataset = { dataset_id: string; name: string };

const searchResult = (datasets: TestDataset[]) =>
  ({
    data: datasets,
    isLoading: false,
    isFetching: false,
    fetchNextPage: jest.fn(),
    hasNextPage: false,
    refetch: jest.fn(),
    error: null,
  }) as any;

const setDatasets = (datasets: TestDataset[]) => {
  mockSearchDatasets.mockReturnValue(searchResult(datasets));
};

const renderModal = ({
  messages = CONVERSATION,
  variables = {},
  linkedPrompt = null,
}: {
  messages?: ConversationMessage[];
  variables?: Record<string, string>;
  linkedPrompt?: { name: string; version: string } | null;
} = {}) => {
  const onCancel = jest.fn();
  const onAdded = jest.fn();
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <AddToDatasetModal
          visible
          onCancel={onCancel}
          experimentId="exp-1"
          messages={messages}
          variables={variables}
          linkedPrompt={linkedPrompt}
          onAdded={onAdded}
        />
      </DesignSystemProvider>
    </IntlProvider>,
  );
  return { onCancel, onAdded };
};

const selectDatasetRow = async (name: RegExp) => {
  // Scoped to the dataset row so it can't pick up the expected-response opt-in elsewhere in the modal.
  const row = await screen.findByRole('row', { name });
  await userEvent.click(within(row).getByRole('checkbox'));
};

const selectFirstDatasetRow = () => selectDatasetRow(/my dataset/i);

const clickAdd = async (name: RegExp = /add to dataset/i) => {
  const addButton = screen.getByRole('button', { name });
  await waitFor(() => expect(addButton).toBeEnabled());
  await userEvent.click(addButton);
};

describe('AddToDatasetModal', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockCheckMultiturn.mockReturnValue({ data: false, isLoading: false });
    mockUpsert(upsertDatasetRecordsMutationAsync.mockResolvedValue({ inserted_count: 1, updated_count: 0 }));
    setDatasets([{ dataset_id: 'd1', name: 'My Dataset' }]);
  });

  it('renders the search input, dataset table and create button, with Add disabled until a dataset is selected', async () => {
    renderModal();

    expect(await screen.findByText('Add to evaluation datasets')).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/search by dataset name/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /create dataset/i })).toBeInTheDocument();
    expect(screen.getByText('My Dataset')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /add to dataset/i })).toBeDisabled();
  });

  it('has no select-all checkbox in the dataset table header', async () => {
    renderModal();

    const [headerRow] = await screen.findAllByRole('row');
    expect(within(headerRow).queryByRole('checkbox')).not.toBeInTheDocument();
  });

  it('opts in and defaults the expected response to the latest reply', async () => {
    renderModal();

    expect(await screen.findByRole('checkbox', { name: /add expected response/i })).toBeChecked();
    expect(screen.getByPlaceholderText(/the reference answer to score responses against/i)).toHaveValue(
      'MLflow is an ML platform.',
    );
  });

  it('upserts the record into the selected dataset and reports success', async () => {
    const { onAdded } = renderModal();

    await selectFirstDatasetRow();
    await clickAdd();

    expect(upsertDatasetRecordsMutationAsync).toHaveBeenCalledWith({
      datasetId: 'd1',
      records: JSON.stringify([
        {
          inputs: { messages: [{ role: 'user', content: 'What is MLflow?' }] },
          expectations: { expected_response: 'MLflow is an ML platform.' },
          source: HUMAN_SOURCE,
        },
      ]),
    });
    await waitFor(() =>
      expect(onAdded).toHaveBeenCalledWith({ datasets: [{ datasetId: 'd1', name: 'My Dataset', updated: false }] }),
    );
    expect(invalidateAfterUpsert).toHaveBeenCalledWith(['d1']);
  });

  it('reports that the existing record was updated when the dataset already had these inputs', async () => {
    mockUpsert(upsertDatasetRecordsMutationAsync.mockResolvedValue({ inserted_count: 0, updated_count: 1 }));
    const { onAdded } = renderModal();

    await selectFirstDatasetRow();
    await clickAdd();

    await waitFor(() =>
      expect(onAdded).toHaveBeenCalledWith({ datasets: [{ datasetId: 'd1', name: 'My Dataset', updated: true }] }),
    );
  });

  it('links the record to the registry prompt version loaded in the playground', async () => {
    renderModal({ linkedPrompt: { name: 'qa-prompt', version: '3' } });

    await selectFirstDatasetRow();
    await clickAdd();

    const [record] = JSON.parse(upsertDatasetRecordsMutationAsync.mock.calls[0][0].records);
    expect(record.tags).toEqual({ 'mlflow.linkedPrompts': JSON.stringify([{ name: 'qa-prompt', version: '3' }]) });
  });

  it('omits expectations when the expected-response opt-in is unchecked', async () => {
    renderModal();

    await userEvent.click(await screen.findByRole('checkbox', { name: /add expected response/i }));
    expect(screen.queryByLabelText(/the reference answer to score responses against/i)).not.toBeInTheDocument();

    await selectFirstDatasetRow();
    await clickAdd();

    expect(upsertDatasetRecordsMutationAsync).toHaveBeenCalledWith({
      datasetId: 'd1',
      records: JSON.stringify([
        { inputs: { messages: [{ role: 'user', content: 'What is MLflow?' }] }, source: HUMAN_SOURCE },
      ]),
    });
  });

  it('leaves the expected-response opt-in unchecked when the prompt has not been run', async () => {
    renderModal({ messages: [{ role: 'user', content: 'What is MLflow?' }] });

    expect(await screen.findByRole('checkbox', { name: /add expected response/i })).not.toBeChecked();
    expect(screen.queryByPlaceholderText(/the reference answer to score responses against/i)).not.toBeInTheDocument();
  });

  it('does not pre-fill a reply generated for different variable values, and says why', async () => {
    renderModal({
      messages: [
        { role: 'user', content: 'What is the capital of {{ country }}?' },
        {
          role: 'assistant',
          content: 'Paris',
          requestMessages: [{ role: 'user', content: 'What is the capital of France?' }],
        },
        { role: 'user', content: '' },
      ],
      variables: { country: 'Japan' },
    });

    expect(await screen.findByRole('checkbox', { name: /add expected response/i })).not.toBeChecked();
    expect(screen.getByText(/the prompt changed after the last run/i)).toBeInTheDocument();
    expect(screen.getByText('What is the capital of Japan?')).toBeInTheDocument();
  });

  it('warns and blocks adding while a prompt variable has no value', async () => {
    renderModal({
      messages: [{ role: 'user', content: 'Summarize {{ topic }} for {{ audience }}' }],
      variables: { topic: 'MLflow' },
    });

    expect(await screen.findByText(/provide a value for the variable audience/i)).toBeInTheDocument();
    await selectFirstDatasetRow();
    expect(screen.getByRole('button', { name: /add to dataset/i })).toBeDisabled();
  });

  it('previews an assistant turn that only called a tool by its tool names', async () => {
    renderModal({
      messages: [
        { role: 'user', content: 'Weather in Paris?' },
        {
          role: 'assistant',
          content: '',
          tool_calls: [{ id: 'call-1', type: 'function', function: { name: 'get_weather', arguments: '{}' } }],
          requestMessages: [{ role: 'user', content: 'Weather in Paris?' }],
        },
        { role: 'user', content: 'And in Tokyo?' },
      ],
    });

    expect(await screen.findByText('Tool call: get_weather')).toBeInTheDocument();
  });

  it('keeps a selection made before searching for another dataset', async () => {
    const datasetA = { dataset_id: 'd1', name: 'My Dataset' };
    const datasetB = { dataset_id: 'd2', name: 'Other Dataset' };
    // Stable results per filter, as react-query returns the same data until it refetches.
    const unfiltered = searchResult([datasetA, datasetB]);
    const filtered = searchResult([datasetB]);
    mockSearchDatasets.mockImplementation(({ nameFilter }: { nameFilter?: string }) =>
      nameFilter ? filtered : unfiltered,
    );
    const { onAdded } = renderModal();

    await selectDatasetRow(/my dataset/i);
    const searchInput = screen.getByPlaceholderText(/search by dataset name/i);
    await userEvent.type(searchInput, 'Other');
    // The search runs on antd's onPressEnter, which checks keyCode; user-event doesn't set it.
    fireEvent.keyDown(searchInput, { key: 'Enter', code: 'Enter', keyCode: 13, which: 13 });
    await waitFor(() => expect(screen.queryByText('My Dataset')).not.toBeInTheDocument());
    await selectDatasetRow(/other dataset/i);
    await clickAdd(/add to 2 datasets/i);

    expect(upsertDatasetRecordsMutationAsync.mock.calls.map(([payload]) => payload.datasetId)).toEqual(['d1', 'd2']);
    await waitFor(() =>
      expect(onAdded).toHaveBeenCalledWith({
        datasets: [
          { datasetId: 'd1', name: 'My Dataset', updated: false },
          { datasetId: 'd2', name: 'Other Dataset', updated: false },
        ],
      }),
    );
  });

  it('selects a dataset created from the picker', async () => {
    renderModal();

    await userEvent.click(await screen.findByRole('button', { name: /create dataset/i }));
    await clickAdd();

    expect(upsertDatasetRecordsMutationAsync).toHaveBeenCalledWith(expect.objectContaining({ datasetId: 'd-new' }));
  });

  it('surfaces an error and skips the success path when the upsert fails', async () => {
    mockUpsert(upsertDatasetRecordsMutationAsync.mockRejectedValue(new Error('Dataset is read-only')));
    const { onAdded } = renderModal();

    await selectFirstDatasetRow();
    await clickAdd();

    expect(await screen.findByText('Dataset is read-only')).toBeInTheDocument();
    expect(onAdded).not.toHaveBeenCalled();
  });

  it('shows the empty state when the experiment has no datasets', async () => {
    setDatasets([]);
    renderModal();

    expect(await screen.findByText(/no evaluation datasets found/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /add to dataset/i })).toBeDisabled();
  });

  it('warns and blocks adding when there is no input message', async () => {
    renderModal({ messages: [{ role: 'user', content: '' }] });

    expect(await screen.findByText(/add at least one non-empty message/i)).toBeInTheDocument();
    await selectFirstDatasetRow();
    expect(screen.getByRole('button', { name: /add to dataset/i })).toBeDisabled();
    expect(upsertDatasetRecordsMutationAsync).not.toHaveBeenCalled();
  });

  it('keeps the modal open while the add is in flight, then reports success', async () => {
    let resolveUpsert: ((value: unknown) => void) | undefined;
    mockUpsert(
      upsertDatasetRecordsMutationAsync.mockImplementation(
        () =>
          new Promise((resolve) => {
            resolveUpsert = resolve;
          }),
      ),
    );
    const { onCancel, onAdded } = renderModal();

    await selectFirstDatasetRow();
    await clickAdd();

    // The upsert can't be cancelled once sent, so Cancel stays disabled until it settles.
    const cancelButton = screen.getByRole('button', { name: /cancel/i });
    expect(cancelButton).toBeDisabled();
    await userEvent.click(cancelButton);
    expect(onCancel).not.toHaveBeenCalled();

    resolveUpsert?.({ inserted_count: 1, updated_count: 0 });
    await waitFor(() =>
      expect(onAdded).toHaveBeenCalledWith({ datasets: [{ datasetId: 'd1', name: 'My Dataset', updated: false }] }),
    );
    await waitFor(() => expect(screen.getByRole('button', { name: /cancel/i })).toBeEnabled());
  });

  it('blocks adding to multi-turn datasets with an error alert', async () => {
    mockCheckMultiturn.mockReturnValue({ data: true, isLoading: false });
    renderModal();

    await selectFirstDatasetRow();

    expect(await screen.findByText(/multi-turn datasets is not yet supported/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /add to dataset/i })).toBeDisabled();
    expect(upsertDatasetRecordsMutationAsync).not.toHaveBeenCalled();
  });
});
