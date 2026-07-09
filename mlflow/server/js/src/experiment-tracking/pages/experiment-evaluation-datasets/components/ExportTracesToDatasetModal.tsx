import { Modal, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useCallback, useState } from 'react';
import { getModelTraceId } from '@databricks/web-shared/model-trace-explorer';
import type { ModelTrace } from '@databricks/web-shared/model-trace-explorer';
import { compact } from 'lodash';
import { extractDatasetInfoFromTraces } from '../utils/datasetUtils';
import { useUpsertDatasetRecordsMutation } from '../hooks/useUpsertDatasetRecordsMutation';
import { useFetchTraces } from '../hooks/useFetchTraces';
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

  const traceIds = selectedTraceInfos.map((traceInfo) =>
    // hacky wrap just to get the id, as this util function expects
    // the full trace, which is not available in the trace table
    getModelTraceId({ info: traceInfo, data: { spans: [] } }),
  );
  const { data: traces, isLoading: isLoadingTraces } = useFetchTraces({ traceIds });
  const datasetRowsToExport = extractDatasetInfoFromTraces(compact(traces));

  const [pickerState, setPickerState] = useState<EvaluationDatasetPickerState>(EMPTY_EVALUATION_DATASET_PICKER_STATE);
  const { selectedDatasets, hasMultiturnDataset, isCheckingMultiturn } = pickerState;

  const { upsertDatasetRecordsMutation, isLoading: isUpsertingDatasetRecords } = useUpsertDatasetRecordsMutation({
    onSuccess: () => {
      setVisible(false);
    },
  });

  const handleExport = useCallback(() => {
    Promise.all(
      selectedDatasets.map((dataset) =>
        upsertDatasetRecordsMutation({
          datasetId: dataset.dataset_id,
          records: JSON.stringify(datasetRowsToExport),
        }),
      ),
    );
  }, [selectedDatasets, upsertDatasetRecordsMutation, datasetRowsToExport]);

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
        disabled: isLoadingTraces || selectedDatasets.length === 0 || hasMultiturnDataset || isCheckingMultiturn,
        loading: isUpsertingDatasetRecords,
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
