import { useMemo } from 'react';

import { ChevronDownIcon, DropdownMenu, Typography, useDesignSystemTheme } from '@databricks/design-system';

import type { TraceView } from './hooks/useTraceViews';
import { useTraceViews } from './hooks/useTraceViews';

const RAW_TRACE_VALUE = '__raw_trace__';

export interface TraceViewSelectorProps {
  traceId: string | null;
  activeViewId: string | null;
  onViewChange: (view: TraceView | null) => void;
}

export const TraceViewSelector = ({ traceId, activeViewId, onViewChange }: TraceViewSelectorProps) => {
  const { theme } = useDesignSystemTheme();
  const { data: views, isLoading } = useTraceViews(traceId);

  const { traceViews, experimentViews } = useMemo(() => {
    const traceScoped: TraceView[] = [];
    const experimentScoped: TraceView[] = [];
    for (const view of views ?? []) {
      if (view.trace_id) {
        traceScoped.push(view);
      } else {
        experimentScoped.push(view);
      }
    }
    return { traceViews: traceScoped, experimentViews: experimentScoped };
  }, [views]);

  const hasViews = (views?.length ?? 0) > 0;

  // Don't render if there are no views or still loading with no data
  if (!hasViews && !isLoading) {
    return null;
  }

  if (isLoading && !hasViews) {
    return null;
  }

  const activeView = views?.find((v) => v.view_id === activeViewId);
  const displayLabel = activeView ? activeView.name : 'Raw Trace';

  const handleValueChange = (value: string) => {
    if (value === RAW_TRACE_VALUE) {
      onViewChange(null);
    } else {
      const selected = views?.find((v) => v.view_id === value);
      if (selected) {
        onViewChange(selected);
      }
    }
  };

  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <div
          css={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: theme.spacing.xs,
            cursor: 'pointer',
            padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
            borderRadius: theme.borders.borderRadiusMd,
            border: `1px solid ${theme.colors.border}`,
            backgroundColor: theme.colors.backgroundPrimary,
          }}
        >
          <Typography.Text size="sm" color="secondary">
            {displayLabel}
          </Typography.Text>
          <ChevronDownIcon />
        </div>
      </DropdownMenu.Trigger>
      <DropdownMenu.Content>
        <DropdownMenu.RadioGroup
          componentId="shared.model-trace-explorer.trace-view-selector"
          value={activeViewId ?? RAW_TRACE_VALUE}
          onValueChange={handleValueChange}
        >
          <DropdownMenu.RadioItem value={RAW_TRACE_VALUE}>
            <DropdownMenu.ItemIndicator />
            Raw Trace
          </DropdownMenu.RadioItem>

          {traceViews.length > 0 && (
            <DropdownMenu.Group>
              <DropdownMenu.Label>Trace Views</DropdownMenu.Label>
              {traceViews.map((view) => (
                <DropdownMenu.RadioItem key={view.view_id} value={view.view_id}>
                  <DropdownMenu.ItemIndicator />
                  {view.name}
                </DropdownMenu.RadioItem>
              ))}
            </DropdownMenu.Group>
          )}

          {experimentViews.length > 0 && (
            <DropdownMenu.Group>
              <DropdownMenu.Label>Experiment Views</DropdownMenu.Label>
              {experimentViews.map((view) => (
                <DropdownMenu.RadioItem key={view.view_id} value={view.view_id}>
                  <DropdownMenu.ItemIndicator />
                  {view.name}
                </DropdownMenu.RadioItem>
              ))}
            </DropdownMenu.Group>
          )}
        </DropdownMenu.RadioGroup>
        <DropdownMenu.Arrow />
      </DropdownMenu.Content>
    </DropdownMenu.Root>
  );
};
