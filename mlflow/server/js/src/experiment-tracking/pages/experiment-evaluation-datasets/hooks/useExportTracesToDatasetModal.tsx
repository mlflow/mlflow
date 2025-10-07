import { useCallback, useState } from 'react';
import { ExportTracesToDatasetModal } from '../components/ExportTracesToDatasetModal';
import { ModelTrace } from '@mlflow/mlflow/src/shared/web-shared/model-trace-explorer';

// used to pass the modal from mlflow codebase to genai-traces-table
export const useExportTracesToDatasetModal = ({ experimentId }: { experimentId: string }) => {
  const [visible, setVisible] = useState(false);
  const renderExportTracesToDatasetsModal = useCallback(
    ({ selectedTraceInfos }: { selectedTraceInfos: ModelTrace['info'][] }) => (
      <ExportTracesToDatasetModal
        experimentId={experimentId}
        visible={visible}
        setVisible={setVisible}
        selectedTraceInfos={selectedTraceInfos}
      />
    ),
    [experimentId, visible, setVisible],
  );

  return {
    showExportTracesToDatasetsModal: visible,
    setShowExportTracesToDatasetsModal: setVisible,
    renderExportTracesToDatasetsModal,
  };
};
