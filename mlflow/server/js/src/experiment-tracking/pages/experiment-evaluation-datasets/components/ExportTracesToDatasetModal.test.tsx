import { describe, expect, jest, test, beforeEach } from '@jest/globals';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import Utils from '@mlflow/mlflow/src/common/utils/Utils';
import { ExportTracesToDatasetModal } from './ExportTracesToDatasetModal';
import { useFetchTraces } from '../hooks/useFetchTraces';
import { useSearchEvaluationDatasets } from '../hooks/useSearchEvaluationDatasets';
import { useCheckMultiturnDatasets } from '../hooks/useCheckMultiturnDatasets';
import { useUpsertDatasetRecordsMutation } from '../hooks/useUpsertDatasetRecordsMutation';
import type { EvaluationDataset } from '../types';

jest.mock('../hooks/useFetchTraces', () => ({
  useFetchTraces: jest.fn(),
}));

jest.mock('../hooks/useSearchEvaluationDatasets', () => ({
  useSearchEvaluationDatasets: jest.fn(),
}));

jest.mock('../hooks/useCheckMultiturnDatasets', () => ({
  useCheckMultiturnDatasets: jest.fn(),
}));

jest.mock('../hooks/useUpsertDatasetRecordsMutation', () => ({
  useUpsertDatasetRecordsMutation: jest.fn(),
}));

jest.mock('../hooks/useInfiniteScrollFetch', () => ({
  useInfiniteScrollFetch: () => () => {},
}));

jest.mock('./CreateEvaluationDatasetButton', () => ({
  CreateEvaluationDatasetButton: () => null,
}));

jest.mock('../utils/datasetUtils', () => ({
  extractDatasetInfoFromTraces: jest.fn((traces: unknown[]) => traces),
}));

const dataset: EvaluationDataset = {
  dataset_id: 'ds-1',
  name: 'eval',
  created_time: 0,
  last_update_time: 0,
  created_by: 'user',
  last_updated_by: 'user',
  experiment_ids: ['exp-1'],
};

const secondDataset: EvaluationDataset = {
  ...dataset,
  dataset_id: 'ds-2',
  name: 'eval-2',
};

describe('ExportTracesToDatasetModal', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.mocked(useSearchEvaluationDatasets).mockReturnValue({
      data: [dataset],
      isLoading: false,
      isFetching: false,
      fetchNextPage: jest.fn(),
      hasNextPage: false,
    } as unknown as ReturnType<typeof useSearchEvaluationDatasets>);
    jest.mocked(useCheckMultiturnDatasets).mockReturnValue({
      data: false,
      isLoading: false,
    } as unknown as ReturnType<typeof useCheckMultiturnDatasets>);
  });

  test('upserts records built from the refetch result, not the cached traces', async () => {
    const user = userEvent.setup();
    const cachedRows = [{ inputs: { q: 'stale' }, expectations: {} }];
    const freshRows = [{ inputs: { q: 'fresh' }, expectations: { length: 50 } }];
    const refetch = jest.fn(async (_options?: { throwOnError?: boolean }) => ({ data: freshRows }));
    const upsertAsync = jest.fn(async (_payload: { datasetId: string; records: string }) => ({
      insertedCount: 1,
      updatedCount: 0,
    }));
    const invalidateAfterUpsert = jest.fn();

    jest.mocked(useFetchTraces).mockReturnValue({
      data: cachedRows,
      isLoading: false,
      refetch,
    } as unknown as ReturnType<typeof useFetchTraces>);
    jest.mocked(useUpsertDatasetRecordsMutation).mockReturnValue({
      upsertDatasetRecordsMutationAsync: upsertAsync,
      invalidateAfterUpsert,
      isLoading: false,
    } as unknown as ReturnType<typeof useUpsertDatasetRecordsMutation>);

    renderWithDesignSystem(
      <ExportTracesToDatasetModal
        experimentId="exp-1"
        visible
        setVisible={() => {}}
        selectedTraceInfos={[{ trace_id: 'tr-1', trace_location: {} } as any]}
      />,
    );

    await user.click(screen.getAllByRole('checkbox')[0]);
    await user.click(screen.getByRole('button', { name: 'Export' }));

    await waitFor(() => {
      expect(refetch).toHaveBeenCalledWith({ throwOnError: true });
      expect(upsertAsync).toHaveBeenCalledWith({
        datasetId: 'ds-1',
        records: JSON.stringify(freshRows),
      });
      expect(invalidateAfterUpsert).toHaveBeenCalledTimes(1);
      expect(invalidateAfterUpsert).toHaveBeenCalledWith(['ds-1']);
    });
  });

  test('does not upsert when refetch fails', async () => {
    const user = userEvent.setup();
    const errorSpy = jest.spyOn(Utils, 'displayGlobalErrorNotification').mockImplementation(() => {});
    const setVisible = jest.fn();
    const refetch = jest.fn(async (_options?: { throwOnError?: boolean }) => {
      throw new Error('refetch failed');
    });
    const upsertAsync = jest.fn(async (_payload: { datasetId: string; records: string }) => ({
      insertedCount: 1,
      updatedCount: 0,
    }));
    const invalidateAfterUpsert = jest.fn();

    jest.mocked(useFetchTraces).mockReturnValue({
      data: [{ inputs: { q: 'stale' }, expectations: {} }],
      isLoading: false,
      refetch,
    } as unknown as ReturnType<typeof useFetchTraces>);
    jest.mocked(useUpsertDatasetRecordsMutation).mockReturnValue({
      upsertDatasetRecordsMutationAsync: upsertAsync,
      invalidateAfterUpsert,
      isLoading: false,
    } as unknown as ReturnType<typeof useUpsertDatasetRecordsMutation>);

    renderWithDesignSystem(
      <ExportTracesToDatasetModal
        experimentId="exp-1"
        visible
        setVisible={setVisible}
        selectedTraceInfos={[{ trace_id: 'tr-1', trace_location: {} } as any]}
      />,
    );

    await user.click(screen.getAllByRole('checkbox')[0]);
    await user.click(screen.getByRole('button', { name: 'Export' }));

    await waitFor(() => {
      expect(refetch).toHaveBeenCalledWith({ throwOnError: true });
      expect(errorSpy).toHaveBeenCalledTimes(1);
    });
    expect(upsertAsync).not.toHaveBeenCalled();
    expect(invalidateAfterUpsert).not.toHaveBeenCalled();
    expect(setVisible).not.toHaveBeenCalled();
    errorSpy.mockRestore();
  });

  test('waits for in-flight upserts before showing an error and re-enabling Export', async () => {
    const user = userEvent.setup();
    const errorSpy = jest.spyOn(Utils, 'displayGlobalErrorNotification').mockImplementation(() => {});
    let resolveSlowUpsert: (value: { insertedCount: number; updatedCount: number }) => void = () => {};
    const slowUpsert = new Promise<{ insertedCount: number; updatedCount: number }>((resolve) => {
      resolveSlowUpsert = resolve;
    });
    const invalidateAfterUpsert = jest.fn();
    const upsertAsync = jest.fn(async ({ datasetId }: { datasetId: string; records: string }) => {
      if (datasetId === 'ds-1') {
        throw new Error('first destination failed');
      }
      return slowUpsert;
    });

    jest.mocked(useSearchEvaluationDatasets).mockReturnValue({
      data: [dataset, secondDataset],
      isLoading: false,
      isFetching: false,
      fetchNextPage: jest.fn(),
      hasNextPage: false,
    } as unknown as ReturnType<typeof useSearchEvaluationDatasets>);
    jest.mocked(useFetchTraces).mockReturnValue({
      data: [{ inputs: { q: 'fresh' }, expectations: {} }],
      isLoading: false,
      refetch: jest.fn(async () => ({ data: [{ inputs: { q: 'fresh' }, expectations: {} }] })),
    } as unknown as ReturnType<typeof useFetchTraces>);
    jest.mocked(useUpsertDatasetRecordsMutation).mockReturnValue({
      upsertDatasetRecordsMutationAsync: upsertAsync,
      invalidateAfterUpsert,
      isLoading: false,
    } as unknown as ReturnType<typeof useUpsertDatasetRecordsMutation>);

    renderWithDesignSystem(
      <ExportTracesToDatasetModal
        experimentId="exp-1"
        visible
        setVisible={() => {}}
        selectedTraceInfos={[{ trace_id: 'tr-1', trace_location: {} } as any]}
      />,
    );

    await user.click(screen.getAllByRole('checkbox')[0]);
    await user.click(screen.getByRole('button', { name: 'Export' }));

    await waitFor(() => {
      expect(upsertAsync).toHaveBeenCalledTimes(2);
    });
    expect(errorSpy).not.toHaveBeenCalled();
    expect(invalidateAfterUpsert).not.toHaveBeenCalled();
    expect(
      document.querySelector('[data-component-id="mlflow.export-traces-to-dataset-modal.footer.ok"]'),
    ).toBeDisabled();

    resolveSlowUpsert({ insertedCount: 1, updatedCount: 0 });

    await waitFor(() => {
      expect(errorSpy).toHaveBeenCalledTimes(1);
      expect(invalidateAfterUpsert).toHaveBeenCalledTimes(1);
      expect(invalidateAfterUpsert).toHaveBeenCalledWith(['ds-2']);
    });
    errorSpy.mockRestore();
  });
});
