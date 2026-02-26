import { useState, useCallback, useMemo } from 'react';
import {
  Button,
  ChevronDownIcon,
  FilterIcon,
  Popover,
  XCircleFillIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import type { TargetScope } from '../../types';

export interface BudgetsFilter {
  scopes: TargetScope[];
}

interface BudgetsFilterButtonProps {
  availableScopes: TargetScope[];
  filter: BudgetsFilter;
  onFilterChange: (filter: BudgetsFilter) => void;
}

const SCOPE_LABELS: Record<TargetScope, string> = {
  GLOBAL: 'Global',
  WORKSPACE: 'Workspace',
};

export const BudgetsFilterButton = ({ availableScopes, filter, onFilterChange }: BudgetsFilterButtonProps) => {
  const { theme } = useDesignSystemTheme();
  const { formatMessage } = useIntl();
  const [open, setOpen] = useState(false);

  const activeCount = filter.scopes.length;

  const handleScopeToggle = useCallback(
    (scope: TargetScope) => {
      const newScopes = filter.scopes.includes(scope)
        ? filter.scopes.filter((s) => s !== scope)
        : [...filter.scopes, scope];
      onFilterChange({ scopes: newScopes });
    },
    [filter.scopes, onFilterChange],
  );

  const handleClear = useCallback(() => {
    onFilterChange({ scopes: [] });
  }, [onFilterChange]);

  const buttonLabel = useMemo(() => {
    if (activeCount === 0) {
      return formatMessage({
        defaultMessage: 'Scope',
        description: 'Gateway > Budget filter > Scope filter button label',
      });
    }
    return formatMessage(
      {
        defaultMessage: 'Scope ({count})',
        description: 'Gateway > Budget filter > Scope filter button label with count',
      },
      { count: activeCount },
    );
  }, [activeCount, formatMessage]);

  return (
    <Popover.Root componentId="mlflow.gateway.budgets-list.filter-popover" open={open} onOpenChange={setOpen}>
      <Popover.Trigger asChild>
        <Button
          componentId="mlflow.gateway.budgets-list.filter-button"
          type="tertiary"
          icon={activeCount > 0 ? <FilterIcon /> : undefined}
          endIcon={<ChevronDownIcon />}
        >
          {buttonLabel}
        </Button>
      </Popover.Trigger>
      <Popover.Content align="start" css={{ padding: theme.spacing.sm, minWidth: 160 }}>
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          {availableScopes.map((scope) => (
            <label
              key={scope}
              css={{
                display: 'flex',
                alignItems: 'center',
                gap: theme.spacing.sm,
                padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
                cursor: 'pointer',
                borderRadius: theme.borders.borderRadiusSm,
                '&:hover': { backgroundColor: theme.colors.actionDefaultBackgroundHover },
              }}
            >
              <input
                type="checkbox"
                checked={filter.scopes.includes(scope)}
                onChange={() => handleScopeToggle(scope)}
              />
              {SCOPE_LABELS[scope]}
            </label>
          ))}
          {activeCount > 0 && (
            <Button
              componentId="mlflow.gateway.budgets-list.filter-clear"
              type="tertiary"
              size="small"
              icon={<XCircleFillIcon />}
              onClick={handleClear}
              css={{ marginTop: theme.spacing.xs }}
            >
              <FormattedMessage
                defaultMessage="Clear filters"
                description="Gateway > Budget filter > Clear filters button"
              />
            </Button>
          )}
        </div>
      </Popover.Content>
    </Popover.Root>
  );
};
