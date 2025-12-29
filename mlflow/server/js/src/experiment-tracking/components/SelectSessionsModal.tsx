import { Modal, TableSkeleton, TableSkeletonRows } from '@databricks/design-system';
import { useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { useParams } from '../../common/utils/RoutingUtils';
import { TracesV3Logs } from './experiment-page/components/traces-v3/TracesV3Logs';
import { GenAiTraceTableRowSelectionProvider } from '@databricks/web-shared/genai-traces-table/hooks/useGenAiTraceTableRowSelection';
import {
  GenAIChatSessionsTable,
  TRACE_ID_COLUMN_ID,
  TracesTableColumn,
  TracesTableColumnType,
  useSearchMlflowTraces,
} from '@databricks/web-shared/genai-traces-table';
import { INPUTS_COLUMN_ID, RESPONSE_COLUMN_ID } from '@databricks/web-shared/genai-traces-table/hooks/useTableColumns';

export const SelectSessionsModal = ({
  onClose,
  onSuccess,
  initialSessionIdsSelected = [],
}: {
  onClose?: () => void;
  onSuccess?: (traceIds: string[]) => void;
  initialSessionIdsSelected?: string[];
}) => {
  const { experimentId } = useParams();

  const [searchQuery, setSearchQuery] = useState('');

  const [rowSelection, setRowSelection] = useState<Record<string, boolean>>(() =>
    initialSessionIdsSelected.reduce((acc, sessionId) => {
      acc[sessionId] = true;
      return acc;
    }, {} as Record<string, boolean>),
  );

  const handleOk = async () => {
    const selectedSessionIds = Object.entries(rowSelection)
      .filter(([_, isSelected]) => isSelected)
      .map(([traceId]) => traceId);
    onSuccess?.(selectedSessionIds);
  };

  const { data: traceInfos, isLoading } = useSearchMlflowTraces({
    locations: [{ mlflow_experiment: { experiment_id: experimentId ?? '' }, type: 'MLFLOW_EXPERIMENT' as const }],
    pageSize: 500,
    limit: 500,
    disabled: !experimentId,
    // TODO (next PRs): Add time range filters
  });

  if (!experimentId) {
    return null;
  }

  return (
    <Modal
      visible
      title={<FormattedMessage defaultMessage="Select sessions" description="Title for the select sessions modal" />}
      componentId="mlflow.experiment-scorers.form.select-sessions-modal"
      onCancel={onClose}
      css={{ width: '90% !important' }}
      size="wide"
      verticalSizing="maxed_out"
      okText={<FormattedMessage defaultMessage="Select" description="Confirm button in the select sessions modal" />}
      okButtonProps={{
        type: 'primary',
        disabled: Object.values(rowSelection).every((isSelected) => !isSelected),
      }}
      onOk={handleOk}
      cancelText={<FormattedMessage defaultMessage="Cancel" description="Cancel button in the select sessions modal" />}
    >
      <GenAiTraceTableRowSelectionProvider rowSelection={rowSelection} setRowSelection={setRowSelection}>
        {isLoading || !traceInfos ? (
          <TableSkeleton lines={3} />
        ) : (
          <GenAIChatSessionsTable
            experimentId={experimentId}
            traces={traceInfos}
            isLoading={false}
            searchQuery={searchQuery}
            setSearchQuery={setSearchQuery}
            enableRowSelection
            enableLinks={false}
          />
        )}
      </GenAiTraceTableRowSelectionProvider>
    </Modal>
  );
};
