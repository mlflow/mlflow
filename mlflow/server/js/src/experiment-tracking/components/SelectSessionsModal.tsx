import { Empty, Modal } from '@databricks/design-system';
import { useMemo, useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { useParams } from '../../common/utils/RoutingUtils';
import { GenAiTraceTableRowSelectionProvider } from '@databricks/web-shared/genai-traces-table/hooks/useGenAiTraceTableRowSelection';
import { GenAIChatSessionsTable, useSearchMlflowTraces } from '@databricks/web-shared/genai-traces-table';
import { getChatSessionsFilter } from '../pages/experiment-chat-sessions/utils';

const MAX_TRACES_PER_PAGE = 500;

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

  const filters = useMemo(() => getChatSessionsFilter({ sessionId: null }), []);

  const { data: traceInfos, isLoading } = useSearchMlflowTraces({
    locations: [{ mlflow_experiment: { experiment_id: experimentId ?? '' }, type: 'MLFLOW_EXPERIMENT' as const }],
    pageSize: MAX_TRACES_PER_PAGE,
    limit: MAX_TRACES_PER_PAGE,
    disabled: !experimentId,
    filters,
    searchQuery,
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
        <GenAIChatSessionsTable
          experimentId={experimentId}
          traces={traceInfos ?? []}
          isLoading={isLoading}
          searchQuery={searchQuery}
          setSearchQuery={setSearchQuery}
          enableRowSelection
          enableLinks={false}
          empty={<EmptySessionsList />}
        />
      </GenAiTraceTableRowSelectionProvider>
    </Modal>
  );
};

const EmptySessionsList = () => {
  return (
    <div css={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
      <Empty
        title={
          <FormattedMessage
            defaultMessage="No sessions found"
            description="Title for the empty sessions list in the select sessions modal"
          />
        }
        description={null}
      />
    </div>
  );
};
