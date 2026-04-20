import { useState, useCallback, useMemo } from 'react';
import {
  Alert,
  Button,
  Checkbox,
  Empty,
  Input,
  Modal,
  Popover,
  SearchIcon,
  Spinner,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  Tag,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayQueryKeys } from '../../hooks/queryKeys';
import { useGuardrailsQuery } from '../../hooks/useGuardrailsQuery';
import { useRemoveGuardrail } from '../../hooks/useRemoveGuardrail';
import { AddGuardrailModal } from './AddGuardrailModal';
import { GuardrailDetailModal } from './GuardrailDetailModal';
import type { GatewayGuardrailConfig, GuardrailStage } from '../../types';

interface GuardrailsTabContentProps {
  endpointName: string;
  endpointId: string;
  experimentId?: string;
}

// ─── Pipeline stage tooltip ──────────────────────────────────────────────────

const PIPELINE_STEPS = ['Request', 'BEFORE', 'LLM', 'AFTER', 'Response'] as const;
// eslint-disable-next-line @databricks/no-const-object-record-string
const STAGE_LABELS: Record<string, string> = { BEFORE: 'Before Guardrails', AFTER: 'After Guardrails' };
// eslint-disable-next-line @databricks/no-const-object-record-string
const STAGE_DESCRIPTIONS: Record<string, string> = {
  BEFORE: 'This guardrail runs before the request reaches the LLM',
  AFTER: 'This guardrail runs after the LLM response is generated',
};

const PlacementTooltipContent = ({ stage }: { stage: GuardrailStage }) => {
  const { theme } = useDesignSystemTheme();
  const activeColor =
    stage === 'BEFORE' ? theme.colors.actionPrimaryBackgroundDefault : theme.colors.textValidationSuccess;

  return (
    <div css={{ minWidth: 360, padding: theme.spacing.xs }}>
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs, marginBottom: theme.spacing.sm }}>
        {PIPELINE_STEPS.map((step, i) => {
          const isActive = step === stage;
          const label = STAGE_LABELS[step] ?? step;
          return (
            <div key={step} css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
              {i > 0 && (
                <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
                  {'>'}
                </Typography.Text>
              )}
              <span
                css={{
                  padding: `2px ${theme.spacing.sm}px`,
                  borderRadius: theme.borders.borderRadiusMd,
                  fontSize: theme.typography.fontSizeSm,
                  fontWeight: isActive ? theme.typography.typographyBoldFontWeight : 'normal',
                  color: isActive ? theme.colors.white : theme.colors.textSecondary,
                  backgroundColor: isActive ? activeColor : 'transparent',
                  border: `1px solid ${isActive ? activeColor : 'transparent'}`,
                  whiteSpace: 'nowrap',
                }}
              >
                {label}
              </span>
            </div>
          );
        })}
      </div>
      <Typography.Text css={{ fontSize: theme.typography.fontSizeSm }}>{STAGE_DESCRIPTIONS[stage]}</Typography.Text>
    </div>
  );
};

// ─── Guardrail row ───────────────────────────────────────────────────────────

const GuardrailRow = ({
  guardrail,
  isSelected,
  onSelectChange,
  onView,
}: {
  guardrail: GatewayGuardrailConfig;
  isSelected: boolean;
  onSelectChange: () => void;
  onView: (g: GatewayGuardrailConfig) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const name = guardrail.guardrail?.name ?? guardrail.guardrail_id;
  const stage = guardrail.guardrail?.stage;
  const action = guardrail.guardrail?.action;
  const stageColor = stage === 'BEFORE' ? 'indigo' : stage === 'AFTER' ? 'teal' : 'default';
  const stageLabel = stage === 'BEFORE' ? 'Before LLM' : stage === 'AFTER' ? 'After LLM' : (stage ?? '—');
  const actionColor = action === 'VALIDATION' ? 'coral' : action === 'SANITIZATION' ? 'purple' : 'default';
  const actionLabel = action === 'VALIDATION' ? 'Block' : action === 'SANITIZATION' ? 'Sanitize' : (action ?? '—');

  return (
    <TableRow key={guardrail.guardrail_id}>
      <TableCell css={{ flex: 0, minWidth: 40, maxWidth: 40 }}>
        <Checkbox
          componentId="mlflow.gateway.guardrails.row-checkbox"
          isChecked={isSelected}
          onChange={onSelectChange}
        />
      </TableCell>
      <TableCell css={{ flex: 2 }}>
        <button
          type="button"
          css={{
            background: 'none',
            border: 'none',
            padding: 0,
            cursor: 'pointer',
            fontWeight: theme.typography.typographyBoldFontWeight,
            color: theme.colors.actionPrimaryBackgroundDefault,
            fontSize: 'inherit',
            '&:hover': { textDecoration: 'underline' },
          }}
          onClick={() => onView(guardrail)}
        >
          {name}
        </button>
      </TableCell>
      <TableCell css={{ flex: 1 }}>
        {stage === 'BEFORE' || stage === 'AFTER' ? (
          <Popover.Root componentId="mlflow.gateway.guardrails.placement-popover">
            <Popover.Trigger asChild>
              <span css={{ cursor: 'default', display: 'inline-block' }}>
                <Tag componentId="mlflow.gateway.guardrails.stage-tag" color={stageColor}>
                  {stageLabel}
                </Tag>
              </span>
            </Popover.Trigger>
            <Popover.Content side="bottom" align="start">
              <PlacementTooltipContent stage={stage} />
            </Popover.Content>
          </Popover.Root>
        ) : (
          <Tag componentId="mlflow.gateway.guardrails.stage-tag" color={stageColor}>
            {stageLabel}
          </Tag>
        )}
      </TableCell>
      <TableCell css={{ flex: 1 }}>
        <Tag componentId="mlflow.gateway.guardrails.action-tag" color={actionColor}>
          {actionLabel}
        </Tag>
      </TableCell>
    </TableRow>
  );
};

// ─── Main component ──────────────────────────────────────────────────────────

export const GuardrailsTabContent = ({ endpointName, endpointId, experimentId }: GuardrailsTabContentProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const queryClient = useQueryClient();
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [detailGuardrail, setDetailGuardrail] = useState<GatewayGuardrailConfig | null>(null);
  const [search, setSearch] = useState('');
  const [rowSelection, setRowSelection] = useState<Record<string, boolean>>({});
  const [deleteModalGuardrails, setDeleteModalGuardrails] = useState<GatewayGuardrailConfig[]>([]);
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const { data: serverGuardrails, isLoading, error } = useGuardrailsQuery(endpointId);
  const { mutateAsync: removeGuardrail } = useRemoveGuardrail();

  const filteredGuardrails = useMemo(() => {
    return serverGuardrails.filter((g) => {
      if (!search.trim()) return true;
      const name = (g.guardrail?.name ?? g.guardrail_id).toLowerCase();
      return name.includes(search.trim().toLowerCase());
    });
  }, [serverGuardrails, search]);

  const selectedGuardrails = useMemo(
    () => filteredGuardrails.filter((g) => rowSelection[g.guardrail_id]),
    [filteredGuardrails, rowSelection],
  );
  const selectedCount = selectedGuardrails.length;
  const allSelected = filteredGuardrails.length > 0 && selectedCount === filteredGuardrails.length;
  const someSelected = selectedCount > 0 && !allSelected;

  const handleSelectAll = useCallback(() => {
    if (allSelected) {
      setRowSelection({});
    } else {
      setRowSelection(Object.fromEntries(filteredGuardrails.map((g) => [g.guardrail_id, true])));
    }
  }, [allSelected, filteredGuardrails]);

  const handleSelectRow = useCallback((guardrailId: string) => {
    setRowSelection((prev) => ({ ...prev, [guardrailId]: !prev[guardrailId] }));
  }, []);

  const handleDeleteClick = useCallback(() => {
    setDeleteError(null);
    setDeleteModalGuardrails(selectedGuardrails);
  }, [selectedGuardrails]);

  const handleConfirmDelete = useCallback(async () => {
    setIsDeleting(true);
    setDeleteError(null);
    try {
      await Promise.all(
        deleteModalGuardrails.map((g) => removeGuardrail({ endpoint_id: endpointId, guardrail_id: g.guardrail_id })),
      );
      setRowSelection({});
      setDeleteModalGuardrails([]);
    } catch {
      setDeleteError(
        intl.formatMessage({
          defaultMessage: 'Failed to remove one or more guardrails. Please try again.',
          description: 'Error when bulk guardrail removal fails',
        }),
      );
    } finally {
      setIsDeleting(false);
    }
  }, [deleteModalGuardrails, removeGuardrail, endpointId, intl]);

  const handleDetailDelete = useCallback(
    (guardrailId: string) => {
      setDeleteError(null);
      setDetailGuardrail(null);
      const toDelete = serverGuardrails.find((g) => g.guardrail_id === guardrailId);
      if (toDelete) setDeleteModalGuardrails([toDelete]);
    },
    [serverGuardrails],
  );

  if (isLoading) {
    return (
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, padding: theme.spacing.md }}>
        <Spinner size="small" />
        <FormattedMessage defaultMessage="Loading guardrails..." description="Loading guardrails" />
      </div>
    );
  }

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      {error && (
        <Alert componentId="mlflow.gateway.guardrails.error" type="error" message={error.message} closable={false} />
      )}

      {/* Toolbar */}
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <Input
          componentId="mlflow.gateway.guardrails.search"
          prefix={<SearchIcon />}
          placeholder={intl.formatMessage({
            defaultMessage: 'Search guardrails',
            description: 'Search guardrails placeholder',
          })}
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          allowClear
          css={{ maxWidth: 300 }}
        />
        <div css={{ marginLeft: 'auto', display: 'flex', gap: theme.spacing.sm }}>
          <Button componentId="mlflow.gateway.guardrails.add" type="primary" onClick={() => setIsAddModalOpen(true)}>
            <FormattedMessage defaultMessage="Create Guardrail" description="Create guardrail button" />
          </Button>
          <Button
            componentId="mlflow.gateway.guardrails.delete"
            danger
            disabled={selectedCount === 0}
            onClick={handleDeleteClick}
          >
            <FormattedMessage
              defaultMessage="Delete{count, select, 0 {} other { ({count})}}"
              description="Delete guardrails button with optional count"
              values={{ count: selectedCount }}
            />
          </Button>
        </div>
      </div>

      {/* Table */}
      <Table
        css={{
          minHeight: 'unset',
          borderLeft: `1px solid ${theme.colors.border}`,
          borderRight: `1px solid ${theme.colors.border}`,
          borderTop: `1px solid ${theme.colors.border}`,
          borderBottom: filteredGuardrails.length === 0 ? `1px solid ${theme.colors.border}` : 'none',
          borderRadius: theme.borders.borderRadiusMd,
          overflow: 'hidden',
        }}
      >
        <TableRow isHeader>
          <TableCell css={{ flex: 0, minWidth: 40, maxWidth: 40 }}>
            <Checkbox
              componentId="mlflow.gateway.guardrails.select-all"
              isChecked={someSelected ? null : allSelected}
              onChange={handleSelectAll}
            />
          </TableCell>
          <TableHeader componentId="mlflow.gateway.guardrails.name-header" css={{ flex: 2 }}>
            <FormattedMessage defaultMessage="Name" description="Guardrail name column header" />
          </TableHeader>
          <TableHeader componentId="mlflow.gateway.guardrails.placement-header" css={{ flex: 1 }}>
            <FormattedMessage defaultMessage="Stage" description="Guardrail stage column header" />
          </TableHeader>
          <TableHeader componentId="mlflow.gateway.guardrails.action-header" css={{ flex: 1 }}>
            <FormattedMessage defaultMessage="Action" description="Guardrail action column header" />
          </TableHeader>
        </TableRow>

        {filteredGuardrails.length === 0 ? (
          <div
            css={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              minHeight: 200,
              width: '100%',
              '& > div': {
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center',
              },
            }}
          >
            <Empty
              description={
                search.trim() ? (
                  <FormattedMessage
                    defaultMessage="No guardrails match your search."
                    description="Empty state when search has no results"
                  />
                ) : (
                  <FormattedMessage
                    defaultMessage="No guardrails configured. Add a guardrail to protect your LLM endpoint."
                    description="Empty state for guardrails"
                  />
                )
              }
            />
          </div>
        ) : (
          filteredGuardrails.map((g) => (
            <GuardrailRow
              key={g.guardrail_id}
              guardrail={g}
              isSelected={Boolean(rowSelection[g.guardrail_id])}
              onSelectChange={() => handleSelectRow(g.guardrail_id)}
              onView={setDetailGuardrail}
            />
          ))
        )}
      </Table>

      <AddGuardrailModal
        open={isAddModalOpen}
        onClose={() => setIsAddModalOpen(false)}
        onSuccess={() => queryClient.invalidateQueries(GatewayQueryKeys.guardrails)}
        endpointName={endpointName}
        endpointId={endpointId}
        experimentId={experimentId}
      />

      {/* Bulk remove confirmation */}
      <Modal
        componentId="mlflow.gateway.guardrails.bulk-remove-modal"
        title={intl.formatMessage(
          {
            defaultMessage: 'Delete {count, plural, one {guardrail} other {# guardrails}}',
            description: 'Bulk delete guardrails modal title',
          },
          { count: deleteModalGuardrails.length },
        )}
        visible={deleteModalGuardrails.length > 0}
        onCancel={() => {
          if (!isDeleting) setDeleteModalGuardrails([]);
        }}
        footer={
          <div css={{ display: 'flex', justifyContent: 'flex-end', gap: theme.spacing.sm }}>
            <Button
              componentId="mlflow.gateway.guardrails.bulk-remove-cancel"
              onClick={() => setDeleteModalGuardrails([])}
              disabled={isDeleting}
            >
              <FormattedMessage defaultMessage="Cancel" description="Cancel button" />
            </Button>
            <Button
              componentId="mlflow.gateway.guardrails.bulk-remove-confirm"
              type="primary"
              danger
              onClick={handleConfirmDelete}
              loading={isDeleting}
            >
              <FormattedMessage defaultMessage="Delete" description="Confirm delete button" />
            </Button>
          </div>
        }
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          {deleteError && (
            <Alert
              componentId="mlflow.gateway.guardrails.bulk-remove-error"
              type="error"
              message={deleteError}
              closable={false}
            />
          )}
          <Typography.Text>
            <FormattedMessage
              defaultMessage="Are you sure you want to remove {count, plural, one {this guardrail} other {these # guardrails}}?"
              description="Bulk remove guardrails confirmation message"
              values={{ count: deleteModalGuardrails.length }}
            />
          </Typography.Text>
          {deleteModalGuardrails.length > 1 && (
            <div
              css={{
                maxHeight: 200,
                overflowY: 'auto',
                display: 'flex',
                flexDirection: 'column',
                gap: theme.spacing.xs,
              }}
            >
              {deleteModalGuardrails.map((g) => (
                <Typography.Text key={g.guardrail_id} bold>
                  {g.guardrail?.name ?? g.guardrail_id}
                </Typography.Text>
              ))}
            </div>
          )}
        </div>
      </Modal>

      <GuardrailDetailModal
        open={!!detailGuardrail}
        onClose={() => setDetailGuardrail(null)}
        onDelete={handleDetailDelete}
        onSuccess={() => queryClient.invalidateQueries(GatewayQueryKeys.guardrails)}
        endpointId={endpointId}
        experimentId={experimentId}
        guardrailConfig={detailGuardrail}
      />
    </div>
  );
};
