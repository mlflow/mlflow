import { useState, useCallback } from 'react';
import {
  Alert,
  Button,
  Empty,
  Input,
  Popover,
  PlusIcon,
  SearchIcon,
  Spinner,
  Tag,
  TrashIcon,
  Typography,
  VisibleIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useGuardrailsQuery } from '../../hooks/useGuardrailsQuery';
import { useRemoveGuardrail } from '../../hooks/useRemoveGuardrail';
import { AddGuardrailModal } from './AddGuardrailModal';
import { DeleteConfirmationModal } from '../common/DeleteConfirmationModal';
import type { GatewayGuardrailConfig, GuardrailStage } from '../../types';

interface GuardrailsTabContentProps {
  endpointName: string;
  endpointId: string;
  experimentId?: string;
}

// ─── Pipeline stage tooltip ──────────────────────────────────────────────────

const PIPELINE_STEPS = ['Request', 'BEFORE', 'LLM', 'AFTER', 'Response'] as const;
const STAGE_LABELS: Record<string, string> = { BEFORE: 'Input Guardrails', AFTER: 'Output Guardrails' };
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
      {/* Pipeline flow */}
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
                  color: isActive ? '#fff' : theme.colors.textSecondary,
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

      {/* Description */}
      <Typography.Text css={{ fontSize: theme.typography.fontSizeSm }}>{STAGE_DESCRIPTIONS[stage]}</Typography.Text>
    </div>
  );
};

// ─── Column header ───────────────────────────────────────────────────────────

const ColumnHeader = ({ children }: { children: React.ReactNode }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <Typography.Text
      color="secondary"
      css={{
        fontSize: theme.typography.fontSizeSm,
        fontWeight: theme.typography.typographyBoldFontWeight,
        textTransform: 'uppercase',
        letterSpacing: '0.05em',
      }}
    >
      {children}
    </Typography.Text>
  );
};

// ─── Guardrail row ───────────────────────────────────────────────────────────

const GuardrailRow = ({
  guardrail,
  onView,
  onDeleteClick,
  isRemoving,
  isViewDisabled,
}: {
  guardrail: GatewayGuardrailConfig;
  onView: (guardrail: GatewayGuardrailConfig) => void;
  onDeleteClick: (guardrail: GatewayGuardrailConfig) => void;
  isRemoving: boolean;
  isViewDisabled?: boolean;
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
    <div
      css={{
        display: 'grid',
        gridTemplateColumns: '1fr 140px 120px 36px 36px',
        alignItems: 'center',
        gap: theme.spacing.md,
        padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
        borderBottom: `1px solid ${theme.colors.border}`,
        '&:last-child': { borderBottom: 'none' },
        '&:hover': { backgroundColor: theme.colors.actionDefaultBackgroundHover },
      }}
    >
      {/* Name */}
      <div css={{ display: 'flex', flexDirection: 'column', gap: 2, minWidth: 0 }}>
        <Typography.Text bold css={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {name}
        </Typography.Text>
      </div>

      {/* Placement badge — with pipeline popover on hover */}
      <div>
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
      </div>

      {/* Action badge */}
      <div>
        <Tag componentId="mlflow.gateway.guardrails.action-tag" color={actionColor}>
          {actionLabel}
        </Tag>
      </div>

      {/* View / edit icon */}
      <div css={{ display: 'flex', justifyContent: 'center' }}>
        <Button
          componentId="mlflow.gateway.guardrails.view"
          icon={<VisibleIcon />}
          size="small"
          type="tertiary"
          onClick={() => onView(guardrail)}
          disabled={isViewDisabled}
          aria-label="View and edit guardrail"
        />
      </div>

      {/* Delete icon */}
      <div css={{ display: 'flex', justifyContent: 'center' }}>
        <Button
          componentId="mlflow.gateway.guardrails.remove"
          icon={<TrashIcon />}
          size="small"
          type="tertiary"
          onClick={() => onDeleteClick(guardrail)}
          loading={isRemoving}
          danger
          aria-label="Remove guardrail"
        />
      </div>
    </div>
  );
};

// ─── Main component ──────────────────────────────────────────────────────────

export const GuardrailsTabContent = ({ endpointName, endpointId, experimentId }: GuardrailsTabContentProps) => {
  const { theme } = useDesignSystemTheme();
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [search, setSearch] = useState('');
  const [pendingDelete, setPendingDelete] = useState<GatewayGuardrailConfig | null>(null);

  const { data: serverGuardrails, isLoading, error } = useGuardrailsQuery(endpointId);
  const { mutateAsync: removeGuardrail, isLoading: isRemoving } = useRemoveGuardrail();

  const handleConfirmDelete = useCallback(async () => {
    if (!pendingDelete) return;
    setPendingDelete(null);
    await removeGuardrail({ endpoint_id: endpointId, guardrail_id: pendingDelete.guardrail_id });
  }, [pendingDelete, removeGuardrail, endpointId]);

  const handleView = useCallback((_guardrail: GatewayGuardrailConfig) => {
    // Detail modal is wired in follow-up PR (7d)
  }, []);

  const handleAdd = useCallback(() => {
    setIsModalOpen(true);
  }, []);

  const handleModalClose = useCallback(() => {
    setIsModalOpen(false);
  }, []);

  const filteredGuardrails = serverGuardrails.filter((g) => {
    if (!search.trim()) return true;
    const name = (g.guardrail?.name ?? g.guardrail_id).toLowerCase();
    return name.includes(search.trim().toLowerCase());
  });

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

      {/* Header: search bar + add button */}
      <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: theme.spacing.md }}>
        <Input
          componentId="mlflow.gateway.guardrails.search"
          prefix={<SearchIcon />}
          placeholder="Search guardrails"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          allowClear
          css={{ maxWidth: 320 }}
        />
        <Button componentId="mlflow.gateway.guardrails.add" type="primary" icon={<PlusIcon />} onClick={handleAdd}>
          <FormattedMessage defaultMessage="Add Guardrail" description="Add guardrail button" />
        </Button>
      </div>

      {/* Table */}
      <div
        css={{
          border: `1px solid ${theme.colors.border}`,
          borderRadius: theme.borders.borderRadiusMd,
          overflow: 'hidden',
        }}
      >
        {/* Column headers */}
        <div
          css={{
            display: 'grid',
            gridTemplateColumns: '1fr 140px 120px 36px 36px',
            gap: theme.spacing.md,
            padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
            backgroundColor: theme.colors.backgroundSecondary,
            borderBottom: `1px solid ${theme.colors.border}`,
          }}
        >
          <ColumnHeader>
            <FormattedMessage defaultMessage="Name" description="Guardrail name column header" />
          </ColumnHeader>
          <ColumnHeader>
            <FormattedMessage defaultMessage="Placement" description="Guardrail placement column header" />
          </ColumnHeader>
          <ColumnHeader>
            <FormattedMessage defaultMessage="Action" description="Guardrail action column header" />
          </ColumnHeader>
          <div />
          <div />
        </div>

        {/* Rows or empty state */}
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
              onView={handleView}
              onDeleteClick={setPendingDelete}
              isRemoving={isRemoving}
              isViewDisabled
            />
          ))
        )}
      </div>

      <AddGuardrailModal
        open={isModalOpen}
        onClose={handleModalClose}
        onSuccess={() => {}}
        endpointName={endpointName}
        endpointId={endpointId}
        experimentId={experimentId}
      />

      <DeleteConfirmationModal
        open={!!pendingDelete}
        onClose={() => setPendingDelete(null)}
        onConfirm={handleConfirmDelete}
        title="Remove Guardrail"
        itemName={pendingDelete?.guardrail?.name ?? pendingDelete?.guardrail_id ?? ''}
        itemType="guardrail"
        componentId="mlflow.gateway.guardrails.delete-confirm"
      />
    </div>
  );
};
