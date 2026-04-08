import { useState, useCallback, useEffect } from 'react';
import {
  Alert,
  Button,
  Empty,
  Input,
  PlusIcon,
  SearchIcon,
  Spinner,
  TrashIcon,
  Typography,
  VisibleIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useGuardrailsQuery } from '../../hooks/useGuardrailsQuery';
import { useRemoveGuardrail } from '../../hooks/useRemoveGuardrail';
import { GuardrailModal } from './AddGuardrailModal';
import type { GatewayGuardrailConfig } from '../../types';

interface GuardrailsTabContentProps {
  endpointName: string;
  endpointId: string;
  experimentId?: string;
}

// ─── Badge ──────────────────────────────────────────────────────────────────

const Badge = ({ label, color }: { label: string; color: string }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <span
      css={{
        display: 'inline-block',
        padding: `2px ${theme.spacing.sm}px`,
        borderRadius: theme.borders.borderRadiusMd,
        backgroundColor: `${color}18`,
        color,
        fontSize: theme.typography.fontSizeSm,
        fontWeight: theme.typography.typographyBoldFontWeight,
        border: `1px solid ${color}40`,
        whiteSpace: 'nowrap',
      }}
    >
      {label}
    </span>
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
  onRemove,
  isRemoving,
}: {
  guardrail: GatewayGuardrailConfig;
  onView: (guardrail: GatewayGuardrailConfig) => void;
  onRemove: (id: string) => void;
  isRemoving: boolean;
}) => {
  const { theme } = useDesignSystemTheme();
  const name = guardrail.guardrail?.name ?? guardrail.guardrail_id;
  const stage = guardrail.guardrail?.stage;
  const action = guardrail.guardrail?.action;
  const stageBadge = stage === 'BEFORE' ? 'Before LLM' : stage === 'AFTER' ? 'After LLM' : stage ?? '—';
  const stageColor = stage === 'BEFORE' ? '#1677ff' : stage === 'AFTER' ? '#52c41a' : '#999';
  const actionBadge = action === 'VALIDATION' ? 'Block' : action === 'SANITIZATION' ? 'Sanitize' : action ?? '—';
  const actionColor = action === 'VALIDATION' ? '#cf1322' : action === 'SANITIZATION' ? '#722ed1' : '#999';

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

      {/* Placement badge */}
      <div>
        <Badge label={stageBadge} color={stageColor} />
      </div>

      {/* Action badge */}
      <div>
        <Badge label={actionBadge} color={actionColor} />
      </div>

      {/* View / detail icon */}
      <div css={{ display: 'flex', justifyContent: 'center' }}>
        <Button
          componentId="mlflow.gateway.guardrails.view"
          icon={<VisibleIcon />}
          size="small"
          type="tertiary"
          onClick={() => onView(guardrail)}
          title="View details"
        />
      </div>

      {/* Delete icon */}
      <div css={{ display: 'flex', justifyContent: 'center' }}>
        <Button
          componentId="mlflow.gateway.guardrails.remove"
          icon={<TrashIcon />}
          size="small"
          type="tertiary"
          onClick={() => onRemove(guardrail.guardrail_id)}
          loading={isRemoving}
          danger
          title="Remove guardrail"
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
  const [removingId, setRemovingId] = useState<string | null>(null);

  const { data: serverGuardrails, isLoading, error, refetch } = useGuardrailsQuery(endpointId);
  const { mutateAsync: removeGuardrail } = useRemoveGuardrail();

  // Local guardrails state for optimistic removal
  const [localGuardrails, setLocalGuardrails] = useState<GatewayGuardrailConfig[]>(serverGuardrails);

  useEffect(() => {
    setLocalGuardrails(serverGuardrails);
  }, [serverGuardrails]);

  const handleRemove = useCallback(
    async (guardrailId: string) => {
      setRemovingId(guardrailId);
      setLocalGuardrails((prev) => prev.filter((g) => g.guardrail_id !== guardrailId));
      try {
        await removeGuardrail({ endpoint_id: endpointId, guardrail_id: guardrailId });
      } finally {
        setRemovingId(null);
      }
    },
    [removeGuardrail, endpointId],
  );

  const handleView = useCallback((_guardrail: GatewayGuardrailConfig) => {
    // Detail modal is wired in follow-up PR (7d)
  }, []);

  const handleAdd = useCallback(() => {
    setIsModalOpen(true);
  }, []);

  const handleModalClose = useCallback(() => {
    setIsModalOpen(false);
  }, []);

  const filteredGuardrails = localGuardrails.filter((g) => {
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
              onRemove={handleRemove}
              isRemoving={removingId === g.guardrail_id}
            />
          ))
        )}
      </div>

      <GuardrailModal
        open={isModalOpen}
        onClose={handleModalClose}
        onSuccess={() => refetch()}
        endpointName={endpointName}
        endpointId={endpointId}
        experimentId={experimentId}
      />
    </div>
  );
};
