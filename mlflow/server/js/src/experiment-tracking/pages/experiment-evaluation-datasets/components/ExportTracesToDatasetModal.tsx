import { Modal, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useCallback, useState } from 'react';
import { getModelTraceId } from '@databricks/web-shared/model-trace-explorer';
import type { ModelTrace } from '@databricks/web-shared/model-trace-explorer';
import { compact } from 'lodash';
import { extractDatasetInfoFromTraces } from '../utils/datasetUtils';
import { useUpsertDatasetRecordsMutation } from '../hooks/useUpsertDatasetRecordsMutation';
import { useFetchTraces } from '../hooks/useFetchTraces';
import Utils from '@mlflow/mlflow/src/common/utils/Utils';
import { EMPTY_EVALUATION_DATASET_PICKER_STATE, EvaluationDatasetPicker } from './EvaluationDatasetPicker';
import type { EvaluationDatasetPickerState } from './EvaluationDatasetPicker';

export const ExportTracesToDatasetModal = ({
  experimentId,
  visible,
  setVisible,
  selectedTraceInfos,
}: {
  experimentId: string;
  visible: boolean;
  setVisible: (visible: boolean) => void;
  selectedTraceInfos: ModelTrace['info'][];
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const traceIds = selectedTraceInfos.map((traceInfo) =>
    // hacky wrap just to get the id, as this util function expects
    // the full trace, which is not available in the trace table
    getModelTraceId({ info: traceInfo, data: { spans: [] } }),
  );
  const { isLoading: isLoadingTraces, refetch: refetchTraces } = useFetchTraces({
    traceIds,
  });
  const [isExporting, setIsExporting] = useState(false);

  const [pickerState, setPickerState] = useState<EvaluationDatasetPickerState>(EMPTY_EVALUATION_DATASET_PICKER_STATE);
  const { selectedDatasets, hasMultiturnDataset, isCheckingMultiturn } = pickerState;

  const {
    upsertDatasetRecordsMutationAsync,
    invalidateAfterUpsert,
    isLoading: isUpsertingDatasetRecords,
  } = useUpsertDatasetRecordsMutation();

  const handleExport = useCallback(async () => {
    setIsExporting(true);
    try {
      // This modal stays mounted, so FETCH_TRACES can be from before the user
      // added expectations. Refetch at export time so the dataset snapshot
      // includes current assessments.
      const { data: freshTraces } = await refetchTraces({ throwOnError: true });
      const datasetRowsToExport = extractDatasetInfoFromTraces(compact(freshTraces));
      const results = await Promise.allSettled(
        selectedDatasets.map((dataset) =>
          upsertDatasetRecordsMutationAsync({
            datasetId: dataset.dataset_id,
            records: JSON.stringify(datasetRowsToExport),
          }),
        ),
      );
      const succeededDatasetIds = selectedDatasets.flatMap((dataset, index) =>
        results[index].status === 'fulfilled' ? [dataset.dataset_id] : [],
      );
      invalidateAfterUpsert(succeededDatasetIds);
      if (results.some((result) => result.status === 'rejected')) {
        throw new Error('Failed to add traces to datasets.');
      }
      setVisible(false);
    } catch {
      Utils.displayGlobalErrorNotification(
        intl.formatMessage({
          defaultMessage: 'Failed to add traces to datasets.',
          description: 'Error toast when adding traces to evaluation datasets fails',
        }),
      );
    } finally {
      setIsExporting(false);
    }
  }, [selectedDatasets, upsertDatasetRecordsMutationAsync, invalidateAfterUpsert, refetchTraces, intl, setVisible]);

  return (
    <Modal
      componentId="mlflow.export-traces-to-dataset-modal"
      visible={visible}
      onCancel={() => setVisible(false)}
      okText={
        <FormattedMessage
          defaultMessage="{count, plural, =0 {Add to dataset} one {Add to dataset} other {Add to # datasets}}"
          description="Confirm-button label on the add-to-evaluation-datasets modal, reflecting how many datasets are selected"
          values={{ count: selectedDatasets.length }}
        />
      }
      okButtonProps={{
        disabled:
          isLoadingTraces || isExporting || selectedDatasets.length === 0 || hasMultiturnDataset || isCheckingMultiturn,
        loading: isUpsertingDatasetRecords || isExporting,
      }}
      onOk={handleExport}
      title={
        <FormattedMessage
          defaultMessage="Add to evaluation datasets"
          description="Title of the add-to-evaluation-datasets modal"
        />
      }
      zIndex={theme.options.zIndexBase + 10}
    >
      <div css={{ height: '500px', overflow: 'hidden' }}>
        <EvaluationDatasetPicker
          experimentId={experimentId}
          onStateChange={setPickerState}
          isLoadingExternal={isLoadingTraces}
        />
      </div>
    </Modal>
  );
};
