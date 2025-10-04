import { useState } from 'react';
import { ExportTracesToDatasetModal } from '../components/ExportTracesToDatasetModal';
import { ModelTrace } from '@mlflow/mlflow/src/shared/web-shared/model-trace-explorer';

export const useExportTracesToDatasetModal = ({ experimentId }: { experimentId: string }) => {
  const [visible, setVisible] = useState(false);

  return {
    showExportTracesToDatasetsModal: visible,
    setShowExportTracesToDatasetsModal: setVisible,
    renderExportTracesToDatasetsModal: ({ selectedTraceInfos }: { selectedTraceInfos: ModelTrace['info'][] }) => (
      <ExportTracesToDatasetModal
        experimentId={experimentId}
        visible={visible}
        setVisible={setVisible}
        selectedTraceInfos={selectedTraceInfos}
      />
    ),
  };
};
