import { Modal } from '@databricks/design-system';
import { useMemo, useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { useParams } from '../../common/utils/RoutingUtils';
import { TracesV3Logs } from './experiment-page/components/traces-v3/TracesV3Logs';
import { GenAiTraceTableRowSelectionProvider } from '../../shared/web-shared/genai-traces-table/hooks/useGenAiTraceTableRowSelection';
import { TracesTableColumn } from '../../shared/web-shared/genai-traces-table';

export const SelectTracesModal = ({
  onClose,
  onSuccess,
  maxTraceCount,
}: {
  onClose?: () => void;
  onSuccess?: (traceIds: string[]) => void;
  maxTraceCount?: number;
  customDefaultSelectedColumns?: (column: TracesTableColumn) => boolean;
}) => {
  const { experimentId } = useParams();

  const [rowSelection, setRowSelection] = useState<Record<string, boolean>>({});

  const handleOk = async () => {
    const selectedTraceIds = Object.entries(rowSelection)
      .filter(([_, isSelected]) => isSelected)
      .map(([traceId]) => traceId);
    onSuccess?.(selectedTraceIds);
  };

  const isMaxTraceCountReached = useMemo(() => {
    if (!maxTraceCount) {
      return false;
    }

    return Object.values(rowSelection).filter((isSelected) => isSelected).length > maxTraceCount;
  }, [maxTraceCount, rowSelection]);

  if (!experimentId) {
    return null;
  }

  return (
    <Modal
      visible
      title={<FormattedMessage defaultMessage="Select traces" description="Title for the select traces modal" />}
      componentId="mlflow.experiment-scorers.form.select-traces-modal"
      onCancel={onClose}
      css={{ width: '90% !important' }}
      size="wide"
      verticalSizing="maxed_out"
      okText={<FormattedMessage defaultMessage="Select" description="Confirm button in the select traces modal" />}
      okButtonProps={{
        type: 'primary',
        disabled: Object.values(rowSelection).every((isSelected) => !isSelected) || isMaxTraceCountReached,
      }}
      onOk={handleOk}
      cancelText={<FormattedMessage defaultMessage="Cancel" description="Cancel button in the select traces modal" />}
    >
      <GenAiTraceTableRowSelectionProvider rowSelection={rowSelection} setRowSelection={setRowSelection}>
        <TracesV3Logs endpointName="" experimentId={experimentId} />
      </GenAiTraceTableRowSelectionProvider>
    </Modal>
  );
};
