import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { PointerEventsCheckLevel } from '@testing-library/user-event';
import { render, screen, waitFor, within } from '@testing-library/react';
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
// The create-dataset button opens its own modal + mutation; stub it to keep this test focused.
jest.mock('../../experiment-evaluation-datasets/components/CreateEvaluationDatasetButton', () => ({
  CreateEvaluationDatasetButton: () => <button type="button">Create dataset</button>,
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

const CONVERSATION: ConversationMessage[] = [
  { role: 'user', content: 'What is MLflow?' },
  { role: 'assistant', content: 'MLflow is an ML platform.' },
  { role: 'user', content: '' },
];

const setDatasets = (datasets: Array<{ dataset_id: string; name: string }>) => {
  mockSearchDatasets.mockReturnValue({
    data: datasets,
    isLoading: false,
    isFetching: false,
    fetchNextPage: jest.fn(),
    hasNextPage: false,
    refetch: jest.fn(),
    error: null,
  } as any);
};

const renderModal = ({
  messages = CONVERSATION,
  variables = {},
}: { messages?: ConversationMessage[]; variables?: Record<string, string> } = {}) => {
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
          onAdded={onAdded}
        />
      </DesignSystemProvider>
    </IntlProvider>,
  );
  return { onCancel, onAdded };
};

const selectFirstDatasetRow = async () => {
  // Scoped to the dataset row so it can't pick up the header select-all or the
  // expected-response opt-in checkbox elsewhere in the modal.
  const row = await screen.findByRole('row', { name: /my dataset/i });
  await userEvent.click(within(row).getByRole('checkbox'));
};

describe('AddToDatasetModal', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockCheckMultiturn.mockReturnValue({ data: false, isLoading: false });
    mockUpsert(upsertDatasetRecordsMutationAsync.mockResolvedValue({ insertedCount: 1, updatedCount: 0 }));
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

  it('opts in and defaults the expected response to the latest assistant reply', async () => {
    renderModal();

    expect(await screen.findByRole('checkbox', { name: /add expected response/i })).toBeChecked();
    expect(screen.getByPlaceholderText(/the reference answer to score responses against/i)).toHaveValue(
      'MLflow is an ML platform.',
    );
  });

  it('upserts the record into the selected dataset and reports success', async () => {
    const { onAdded } = renderModal();

    await selectFirstDatasetRow();

    const addButton = screen.getByRole('button', { name: /add to dataset/i });
    await waitFor(() => expect(addButton).toBeEnabled());
    await userEvent.click(addButton);

    expect(upsertDatasetRecordsMutationAsync).toHaveBeenCalledWith({
      datasetId: 'd1',
      records: JSON.stringify([
        {
          inputs: { messages: [{ role: 'user', content: 'What is MLflow?' }] },
          expectations: { expected_response: 'MLflow is an ML platform.' },
        },
      ]),
    });
    await waitFor(() => expect(onAdded).toHaveBeenCalledWith({ datasetNames: ['My Dataset'] }));
    expect(invalidateAfterUpsert).toHaveBeenCalledWith(['d1']);
  });

  it('omits expectations when the expected-response opt-in is unchecked', async () => {
    renderModal();

    await userEvent.click(await screen.findByRole('checkbox', { name: /add expected response/i }));
    expect(screen.queryByLabelText(/the reference answer to score responses against/i)).not.toBeInTheDocument();

    await selectFirstDatasetRow();
    const addButton = screen.getByRole('button', { name: /add to dataset/i });
    await waitFor(() => expect(addButton).toBeEnabled());
    await userEvent.click(addButton);

    expect(upsertDatasetRecordsMutationAsync).toHaveBeenCalledWith({
      datasetId: 'd1',
      records: JSON.stringify([{ inputs: { messages: [{ role: 'user', content: 'What is MLflow?' }] } }]),
    });
  });

  it('leaves the expected-response opt-in unchecked when the prompt has not been run', async () => {
    renderModal({ messages: [{ role: 'user', content: 'What is MLflow?' }] });

    expect(await screen.findByRole('checkbox', { name: /add expected response/i })).not.toBeChecked();
    expect(screen.queryByPlaceholderText(/the reference answer to score responses against/i)).not.toBeInTheDocument();
  });

  it('surfaces an error and skips the success path when the upsert fails', async () => {
    mockUpsert(upsertDatasetRecordsMutationAsync.mockRejectedValue(new Error('Dataset is read-only')));
    const { onAdded } = renderModal();

    await selectFirstDatasetRow();
    const addButton = screen.getByRole('button', { name: /add to dataset/i });
    await waitFor(() => expect(addButton).toBeEnabled());
    await userEvent.click(addButton);

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

  it('does not report success for an in-flight upsert after the modal was cancelled', async () => {
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
    const addButton = screen.getByRole('button', { name: /add to dataset/i });
    await waitFor(() => expect(addButton).toBeEnabled());
    await userEvent.click(addButton);

    await userEvent.click(screen.getByRole('button', { name: /cancel/i }));
    expect(onCancel).toHaveBeenCalled();

    // The record still lands, but the abandoned interaction must not fire the success path.
    resolveUpsert?.({ insertedCount: 1, updatedCount: 0 });
    await waitFor(() => expect(invalidateAfterUpsert).toHaveBeenCalledWith(['d1']));
    expect(onAdded).not.toHaveBeenCalled();
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
