import {
  Button,
  Checkbox,
  FilterIcon,
  InfoTooltip,
  Popover,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import type { SpanFilterState } from '../ModelTrace.types';
import { getDisplayNameForSpanType, getIconTypeForSpan } from '../ModelTraceExplorer.utils';
import { ModelTraceExplorerIcon } from '../ModelTraceExplorerIcon';

export const TimelineTreeFilterButton = ({
  spanFilterState,
  setSpanFilterState,
}: {
  spanFilterState: SpanFilterState;
  setSpanFilterState: (state: SpanFilterState) => void;
}) => {
  const { theme } = useDesignSystemTheme();

  return (
    <Popover.Root componentId="shared.model-trace-explorer.timeline-tree-filter-popover">
      <Popover.Trigger asChild>
        <Button
          componentId="shared.model-trace-explorer.timeline-tree-filter-button"
          icon={<FilterIcon />}
          size="small"
        >
          <FormattedMessage defaultMessage="Filter" description="Label for the filter button in the trace explorer." />
        </Button>
      </Popover.Trigger>
      <Popover.Content align="start">
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm, paddingBottom: theme.spacing.xs }}>
          <Typography.Text bold>
            <FormattedMessage
              defaultMessage="Filter"
              description="Label for the span filters popover in the trace explorer."
            />
          </Typography.Text>
          <Typography.Text color="secondary">
            <FormattedMessage
              defaultMessage="Span type"
              description="Section label for span type filters in the trace explorer."
            />
          </Typography.Text>
          {Object.entries(spanFilterState.spanTypeDisplayState).map(([spanType, shouldDisplay]) => {
            const icon = <ModelTraceExplorerIcon type={getIconTypeForSpan(spanType)} />;
            return (
              <Checkbox
                key={spanType}
                componentId={`shared.model-trace-explorer.toggle-span-filter_${spanType}-${!shouldDisplay}`}
                style={{ width: '100%' }}
                isChecked={shouldDisplay}
                onChange={() =>
                  setSpanFilterState({
                    ...spanFilterState,
                    spanTypeDisplayState: {
                      ...spanFilterState.spanTypeDisplayState,
                      [spanType]: !shouldDisplay,
                    },
                  })
                }
              >
                {icon}
                <Typography.Text css={{ marginLeft: theme.spacing.xs }}>
                  {getDisplayNameForSpanType(spanType)}
                </Typography.Text>
              </Checkbox>
            );
          })}
          <Typography.Text color="secondary">
            <FormattedMessage
              defaultMessage="Settings"
              description="Section label for filter settings in the trace explorer."
            />
          </Typography.Text>
          <Checkbox
            componentId={`shared.model-trace-explorer.toggle-show-parents_${!spanFilterState.showParents}`}
            style={{ width: '100%' }}
            isChecked={spanFilterState.showParents}
            onChange={() =>
              setSpanFilterState({
                ...spanFilterState,
                showParents: !spanFilterState.showParents,
              })
            }
          >
            <Typography.Text css={{ marginRight: theme.spacing.xs }}>
              <FormattedMessage
                defaultMessage="Show all parent spans"
                description="Checkbox label for a setting that enables showing all parent spans in the trace explorer regardless of filter conditions."
              />
            </Typography.Text>
            <InfoTooltip
              componentId="shared.model-trace-explorer.show-parents-tooltip"
              content={
                <FormattedMessage
                  defaultMessage="Always show parents of matched spans, regardless of filter conditions"
                  description="Tooltip for a span filter setting that enables showing parents of matched spans"
                />
              }
            />
          </Checkbox>
          <Checkbox
            componentId={`shared.model-trace-explorer.toggle-show-parents_${!spanFilterState.showExceptions}`}
            style={{ width: '100%' }}
            isChecked={spanFilterState.showExceptions}
            onChange={() =>
              setSpanFilterState({
                ...spanFilterState,
                showExceptions: !spanFilterState.showExceptions,
              })
            }
          >
            <Typography.Text css={{ marginRight: theme.spacing.xs }}>
              <FormattedMessage
                defaultMessage="Show exceptions"
                description="Checkbox label for a setting that enables showing spans with exceptions in the trace explorer regardless of filter conditions."
              />
            </Typography.Text>
            <InfoTooltip
              componentId="shared.model-trace-explorer.show-exceptions-tooltip"
              content={
                <FormattedMessage
                  defaultMessage="Always show spans with exceptions, regardless of filter conditions"
                  description="Tooltip for a span filter setting that enables showing spans with exceptions"
                />
              }
            />
          </Checkbox>
        </div>
      </Popover.Content>
    </Popover.Root>
  );
};
