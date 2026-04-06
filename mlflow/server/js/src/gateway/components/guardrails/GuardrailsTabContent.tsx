import { useState, useCallback, useMemo, useEffect, useRef } from 'react';
import {
  Alert,
  Button,
  Empty,
  PlusIcon,
  Spinner,
  TrashIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useGuardrailsQuery } from '../../hooks/useGuardrailsQuery';
import { useRemoveGuardrail } from '../../hooks/useRemoveGuardrail';
import { useUpdateGuardrail } from '../../hooks/useUpdateGuardrail';
import { GuardrailModal } from './AddGuardrailModal';
import type { GatewayGuardrailConfig } from '../../types';

interface GuardrailsTabContentProps {
  endpointName: string;
  endpointId: string;
  experimentId?: string;
}

const OPERATION_LABELS: Record<string, string> = {
  VALIDATION: 'Validation',
  SANITIZATION: 'Sanitize',
};

const STAGE_COLORS: Record<string, string> = {
  BEFORE: '#1677ff',
  AFTER: '#52c41a',
};

const OPERATION_COLORS: Record<string, string> = {
  VALIDATION: '#faad14',
  SANITIZATION: '#722ed1',
};

const Badge = ({ label, color }: { label: string; color: string }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <span
      css={{
        display: 'inline-block',
        padding: `1px ${theme.spacing.xs}px`,
        borderRadius: theme.borders.borderRadiusMd,
        backgroundColor: `${color}18`,
        color,
        fontSize: theme.typography.fontSizeSm,
        fontWeight: theme.typography.typographyBoldFontWeight,
        border: `1px solid ${color}40`,
      }}
    >
      {label}
    </span>
  );
};

const GuardrailRow = ({
  guardrail,
  index,
  total,
  onEdit,
  onRemove,
  onMoveUp,
  onMoveDown,
  isRemoving,
  isReordering,
  showReorder,
}: {
  guardrail: GatewayGuardrailConfig;
  index: number;
  total: number;
  onEdit: (guardrail: GatewayGuardrailConfig) => void;
  onRemove: (id: string) => void;
  onMoveUp: (id: string) => void;
  onMoveDown: (id: string) => void;
  isRemoving: boolean;
  isReordering: boolean;
  showReorder: boolean;
}) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: theme.spacing.md,
        borderBottom: `1px solid ${theme.colors.border}`,
        '&:last-child': { borderBottom: 'none' },
        '&:hover': { backgroundColor: theme.colors.tableRowHover },
        cursor: 'pointer',
      }}
      onClick={() => onEdit(guardrail)}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onEdit(guardrail);
        }
      }}
    >
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.md }}>
        {/* Order number — only for mutation guardrails where order matters */}
        {showReorder && (
          <span
            css={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: 24,
              height: 24,
              borderRadius: '50%',
              backgroundColor: theme.colors.actionDefaultBackgroundDefault,
              color: theme.colors.textSecondary,
              fontSize: theme.typography.fontSizeSm,
              fontWeight: theme.typography.typographyBoldFontWeight,
              flexShrink: 0,
            }}
          >
            {index + 1}
          </span>
        )}

        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
            <Typography.Text bold>{guardrail.guardrail?.name ?? guardrail.guardrail_id}</Typography.Text>
            <Badge
              label={OPERATION_LABELS[guardrail.guardrail?.action ?? ''] ?? guardrail.guardrail?.action}
              color={OPERATION_COLORS[guardrail.guardrail?.action ?? ''] ?? '#999'}
            />
          </div>
          {guardrail.endpoint_id ? (
            <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
              Endpoint: {guardrail.endpoint_id}
            </Typography.Text>
          ) : null}
        </div>
      </div>

      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }} onClick={(e) => e.stopPropagation()}>
        {/* Reorder buttons — only for mutation guardrails */}
        {showReorder && (
          <>
            <Button
              componentId="mlflow.gateway.guardrails.move-up"
              size="small"
              onClick={() => onMoveUp(guardrail.guardrail_id)}
              disabled={index === 0 || isReordering}
              css={{ minWidth: 28, padding: '0 4px' }}
            >
              {'\u2191'}
            </Button>
            <Button
              componentId="mlflow.gateway.guardrails.move-down"
              size="small"
              onClick={() => onMoveDown(guardrail.guardrail_id)}
              disabled={index === total - 1 || isReordering}
              css={{ minWidth: 28, padding: '0 4px' }}
            >
              {'\u2193'}
            </Button>
          </>
        )}

        <Button
          componentId="mlflow.gateway.guardrails.remove"
          icon={<TrashIcon />}
          onClick={() => onRemove(guardrail.guardrail_id)}
          loading={isRemoving}
          danger
        />
      </div>
    </div>
  );
};

export const GuardrailsTabContent = ({ endpointName, endpointId, experimentId }: GuardrailsTabContentProps) => {
  const { theme } = useDesignSystemTheme();
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [editingGuardrail, setEditingGuardrail] = useState<GatewayGuardrailConfig | null>(null);
  const { data: serverGuardrails, isLoading, error, refetch } = useGuardrailsQuery(endpointId);
  const { mutateAsync: removeGuardrail } = useRemoveGuardrail();
  const { mutateAsync: updateConfig, isLoading: isReordering } = useUpdateGuardrail();
  const [removingId, setRemovingId] = useState<string | null>(null);

  // Optimistic local state for guardrails — updated immediately on reorder
  const [localGuardrails, setLocalGuardrails] = useState<GatewayGuardrailConfig[]>(serverGuardrails);
  const isOptimistic = useRef(false);

  // Sync local state from server when not in an optimistic update
  useEffect(() => {
    if (!isOptimistic.current) {
      setLocalGuardrails(serverGuardrails);
    }
  }, [serverGuardrails]);

  const guardrails = localGuardrails;

  const handleRemove = useCallback(
    async (guardrailId: string) => {
      setRemovingId(guardrailId);
      // Optimistically remove from local state
      setLocalGuardrails((prev) => prev.filter((g) => g.guardrail_id !== guardrailId));
      try {
        await removeGuardrail({ endpoint_id: endpointId, guardrail_id: guardrailId });
      } finally {
        setRemovingId(null);
      }
    },
    [removeGuardrail, endpointId],
  );

  const handleEdit = useCallback((guardrail: GatewayGuardrailConfig) => {
    setEditingGuardrail(guardrail);
    setIsModalOpen(true);
  }, []);

  const handleAdd = useCallback(() => {
    setEditingGuardrail(null);
    setIsModalOpen(true);
  }, []);

  const handleModalClose = useCallback(() => {
    setIsModalOpen(false);
    setEditingGuardrail(null);
  }, []);

  // Separate pre/post, sort: VALIDATION first (parallel), then MUTATION (sequential, order matters)
  const sortByOperation = (list: GatewayGuardrailConfig[]) =>
    [...list].sort((a, b) => {
      if (a.guardrail?.action === b.guardrail?.action) return 0;
      return a.guardrail?.action === 'VALIDATION' ? -1 : 1;
    });

  const preGuardrails = useMemo(
    () => sortByOperation(guardrails.filter((g) => g.guardrail?.stage === 'BEFORE')),
    [guardrails],
  );
  const postGuardrails = useMemo(
    () => sortByOperation(guardrails.filter((g) => g.guardrail?.stage === 'AFTER')),
    [guardrails],
  );

  // Only mutation guardrails within a hook section can be reordered
  const handleMoveMutation = useCallback(
    async (hook: 'PRE' | 'POST', id: string, direction: 'up' | 'down') => {
      const hookList = hook === 'PRE' ? preGuardrails : postGuardrails;
      const mutationOnly = hookList.filter((g) => g.guardrail?.action === 'SANITIZATION');
      const idx = mutationOnly.findIndex((g) => g.guardrail_id === id);
      if (idx < 0) return;
      const swapIdx = direction === 'up' ? idx - 1 : idx + 1;
      if (swapIdx < 0 || swapIdx >= mutationOnly.length) return;

      const reordered = [...mutationOnly];
      [reordered[idx], reordered[swapIdx]] = [reordered[swapIdx], reordered[idx]];

      // Optimistically update local state immediately
      isOptimistic.current = true;
      setLocalGuardrails((prev) => {
        const updated = [...prev];
        const aIdx = updated.findIndex((g) => g.guardrail_id === reordered[idx]?.guardrail_id);
        const bIdx = updated.findIndex((g) => g.guardrail_id === reordered[swapIdx]?.guardrail_id);
        if (aIdx >= 0 && bIdx >= 0) {
          [updated[aIdx], updated[bIdx]] = [updated[bIdx], updated[aIdx]];
        }
        return updated;
      });

      try {
        await Promise.all(
          reordered.map((g, i) =>
            updateConfig({
              endpoint_id: endpointId,
              guardrail_id: g.guardrail_id,
              execution_order: i + 1,
            }),
          ),
        );
      } catch {
        setLocalGuardrails(serverGuardrails);
      } finally {
        isOptimistic.current = false;
      }
    },
    [updateConfig, endpointId, serverGuardrails, preGuardrails, postGuardrails],
  );

  const handleMoveUpPre = useCallback((id: string) => handleMoveMutation('PRE', id, 'up'), [handleMoveMutation]);
  const handleMoveDownPre = useCallback((id: string) => handleMoveMutation('PRE', id, 'down'), [handleMoveMutation]);
  const handleMoveUpPost = useCallback((id: string) => handleMoveMutation('POST', id, 'up'), [handleMoveMutation]);
  const handleMoveDownPost = useCallback((id: string) => handleMoveMutation('POST', id, 'down'), [handleMoveMutation]);

  if (isLoading) {
    return (
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, padding: theme.spacing.md }}>
        <Spinner size="small" />
        <FormattedMessage defaultMessage="Loading guardrails..." description="Loading guardrails" />
      </div>
    );
  }

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
      {error && (
        <Alert componentId="mlflow.gateway.guardrails.error" type="error" message={error.message} closable={false} />
      )}

      {/* Header with add button */}
      <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <Typography.Title level={3}>
            <FormattedMessage defaultMessage="Guardrails" description="Guardrails section title" />
          </Typography.Title>
          <Typography.Text color="secondary">
            <FormattedMessage
              defaultMessage="Configure safety guardrails that run before or after LLM invocation to validate or modify requests and responses."
              description="Guardrails section description"
            />
          </Typography.Text>
        </div>
        <Button componentId="mlflow.gateway.guardrails.add" type="primary" icon={<PlusIcon />} onClick={handleAdd}>
          <FormattedMessage defaultMessage="Add guardrail" description="Add guardrail button" />
        </Button>
      </div>

      {guardrails.length === 0 ? (
        <div
          css={{
            padding: theme.spacing.lg,
            border: `2px dashed ${theme.colors.actionDefaultBorderDefault}`,
            borderRadius: theme.borders.borderRadiusMd,
            textAlign: 'center',
          }}
        >
          <Empty
            description={
              <FormattedMessage
                defaultMessage="No guardrails configured. Add a guardrail to protect your LLM endpoint."
                description="Empty state for guardrails"
              />
            }
          />
        </div>
      ) : (
        <>
          {/* Execution flow diagram */}
          <div
            css={{
              display: 'flex',
              alignItems: 'center',
              gap: theme.spacing.sm,
              padding: theme.spacing.md,
              backgroundColor: theme.colors.backgroundSecondary,
              borderRadius: theme.borders.borderRadiusMd,
              border: `1px solid ${theme.colors.border}`,
              justifyContent: 'center',
              flexWrap: 'wrap',
            }}
          >
            <FlowStep label="Request" active={false} />
            <FlowArrow />
            <FlowStep
              label={`Pre Guardrails (${preGuardrails.length})`}
              active={preGuardrails.length > 0}
              color={STAGE_COLORS['BEFORE']}
            />
            <FlowArrow />
            <FlowStep label="LLM" active />
            <FlowArrow />
            <FlowStep
              label={`Post Guardrails (${postGuardrails.length})`}
              active={postGuardrails.length > 0}
              color={STAGE_COLORS['AFTER']}
            />
            <FlowArrow />
            <FlowStep label="Response" active={false} />
          </div>

          {/* Pre/Post guardrails side by side */}
          <div css={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: theme.spacing.md }}>
            <GuardrailSection
              title="Pre-invocation"
              count={preGuardrails.length}
              color={STAGE_COLORS['BEFORE']}
              guardrails={preGuardrails}
              onEdit={handleEdit}
              onRemove={handleRemove}
              onMoveUp={handleMoveUpPre}
              onMoveDown={handleMoveDownPre}
              removingId={removingId}
              isReordering={isReordering}
            />
            <GuardrailSection
              title="Post-invocation"
              count={postGuardrails.length}
              color={STAGE_COLORS['AFTER']}
              guardrails={postGuardrails}
              onEdit={handleEdit}
              onRemove={handleRemove}
              onMoveUp={handleMoveUpPost}
              onMoveDown={handleMoveDownPost}
              removingId={removingId}
              isReordering={isReordering}
            />
          </div>
        </>
      )}

      <GuardrailModal
        open={isModalOpen}
        onClose={handleModalClose}
        onSuccess={() => refetch()}
        onDelete={(guardrailId) => {
          handleRemove(guardrailId);
          handleModalClose();
        }}
        endpointName={endpointName}
        endpointId={endpointId}
        editingGuardrail={editingGuardrail}
        experimentId={experimentId}
      />
    </div>
  );
};

// ─── Guardrail section ──────────────────────────────────────────────────────

const GuardrailSection = ({
  title,
  count,
  color,
  guardrails,
  onEdit,
  onRemove,
  onMoveUp,
  onMoveDown,
  removingId,
  isReordering,
}: {
  title: string;
  count: number;
  color: string;
  guardrails: GatewayGuardrailConfig[];
  onEdit: (guardrail: GatewayGuardrailConfig) => void;
  onRemove: (id: string) => void;
  onMoveUp: (id: string) => void;
  onMoveDown: (id: string) => void;
  removingId: string | null;
  isReordering: boolean;
}) => {
  const { theme } = useDesignSystemTheme();

  const validationGuardrails = guardrails.filter((g) => g.guardrail?.action === 'VALIDATION');
  const mutationGuardrails = guardrails.filter((g) => g.guardrail?.action === 'SANITIZATION');

  return (
    <div
      css={{
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        backgroundColor: theme.colors.backgroundSecondary,
        overflow: 'hidden',
      }}
    >
      <div
        css={{
          padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
          borderBottom: `1px solid ${theme.colors.border}`,
          backgroundColor: `${color}08`,
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.sm,
        }}
      >
        <Badge label={title} color={color} />
        <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
          {count} {count === 1 ? 'guardrail' : 'guardrails'}
        </Typography.Text>
      </div>

      {guardrails.length === 0 && (
        <div css={{ padding: theme.spacing.md }}>
          <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
            <FormattedMessage defaultMessage="No guardrails configured." description="Empty guardrail section" />
          </Typography.Text>
        </div>
      )}

      {/* Validation guardrails — run in parallel, no ordering */}
      {validationGuardrails.length > 0 && (
        <>
          <div
            css={{
              padding: `${theme.spacing.xs}px ${theme.spacing.md}px`,
              borderBottom: `1px solid ${theme.colors.border}`,
              backgroundColor: `${OPERATION_COLORS['VALIDATION']}08`,
              display: 'flex',
              alignItems: 'center',
              gap: theme.spacing.sm,
            }}
          >
            <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
              <FormattedMessage
                defaultMessage="Block (parallel)"
                description="Validation sub-header in guardrail section"
              />
            </Typography.Text>
          </div>
          {validationGuardrails.map((g, i) => (
            <GuardrailRow
              key={g.guardrail_id}
              guardrail={g}
              index={i}
              total={validationGuardrails.length}
              onEdit={onEdit}
              onRemove={onRemove}
              onMoveUp={onMoveUp}
              onMoveDown={onMoveDown}
              isRemoving={removingId === g.guardrail_id}
              isReordering={isReordering}
              showReorder={false}
            />
          ))}
        </>
      )}

      {/* Mutation guardrails — run sequentially, order matters */}
      {mutationGuardrails.length > 0 && (
        <>
          <div
            css={{
              padding: `${theme.spacing.xs}px ${theme.spacing.md}px`,
              borderBottom: `1px solid ${theme.colors.border}`,
              backgroundColor: `${OPERATION_COLORS['SANITIZATION']}08`,
              display: 'flex',
              alignItems: 'center',
              gap: theme.spacing.sm,
            }}
          >
            <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
              <FormattedMessage
                defaultMessage="Modify (sequential)"
                description="Modify sub-header in guardrail section"
              />
            </Typography.Text>
          </div>
          {mutationGuardrails.map((g, i) => (
            <GuardrailRow
              key={g.guardrail_id}
              guardrail={g}
              index={i}
              total={mutationGuardrails.length}
              onEdit={onEdit}
              onRemove={onRemove}
              onMoveUp={onMoveUp}
              onMoveDown={onMoveDown}
              isRemoving={removingId === g.guardrail_id}
              isReordering={isReordering}
              showReorder
            />
          ))}
        </>
      )}
    </div>
  );
};

// ─── Flow diagram helper components ─────────────────────────────────────────

const FlowStep = ({ label, active, color }: { label: string; active: boolean; color?: string }) => {
  const { theme } = useDesignSystemTheme();
  const bgColor = active ? (color ? `${color}18` : theme.colors.actionPrimaryBackgroundDefault + '18') : 'transparent';
  const borderColor = active ? (color ?? theme.colors.actionPrimaryBackgroundDefault) : theme.colors.border;
  const textColor = active ? (color ?? theme.colors.textPrimary) : theme.colors.textSecondary;

  return (
    <div
      css={{
        padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
        borderRadius: theme.borders.borderRadiusMd,
        border: `1px solid ${borderColor}`,
        backgroundColor: bgColor,
        color: textColor,
        fontSize: theme.typography.fontSizeSm,
        fontWeight: active ? theme.typography.typographyBoldFontWeight : 'normal',
        whiteSpace: 'nowrap',
      }}
    >
      {label}
    </div>
  );
};

const FlowArrow = () => {
  const { theme } = useDesignSystemTheme();
  return <span css={{ color: theme.colors.textSecondary, fontSize: theme.typography.fontSizeSm }}>{'\u2192'}</span>;
};
