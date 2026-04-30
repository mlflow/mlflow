import {
  ApplyDesignSystemContextOverrides,
  Button,
  Checkbox,
  FilterIcon,
  InfoTooltip,
  Popover,
  SimpleSelect,
  SimpleSelectOption,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import type { SpanFilterState } from '../ModelTrace.types';
import { SpanLogLevel } from '../ModelTrace.types';
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
                componentId="shared.model-trace-explorer.toggle-span-filter"
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
              defaultMessage="Minimum log level"
              description="Section label for the minimum span log level filter in the trace explorer."
            />
          </Typography.Text>
          {/* Bump the design-system z-index so the Select's dropdown portal lands
              above the parent filter Popover instead of being clipped behind it. */}
          <ApplyDesignSystemContextOverrides zIndexBase={2 * theme.options.zIndexBase}>
            <SimpleSelect
              id="shared.model-trace-explorer.span-log-level-filter"
              componentId="shared.model-trace-explorer.span-log-level-filter"
              label="Minimum log level"
              value={String(spanFilterState.minLogLevel)}
              onChange={(e) =>
                setSpanFilterState({
                  ...spanFilterState,
                  minLogLevel: Number(e.target.value) as SpanLogLevel,
                })
              }
            >
              <SimpleSelectOption value={String(SpanLogLevel.DEBUG)}>
                <FormattedMessage
                  defaultMessage="Debug"
                  description="Option in the span log-level filter that shows all spans (DEBUG and above)."
                />
              </SimpleSelectOption>
              <SimpleSelectOption value={String(SpanLogLevel.INFO)}>
                <FormattedMessage
                  defaultMessage="Info"
                  description="Option in the span log-level filter that hides DEBUG-level spans."
                />
              </SimpleSelectOption>
              <SimpleSelectOption value={String(SpanLogLevel.WARNING)}>
                <FormattedMessage
                  defaultMessage="Warning"
                  description="Option in the span log-level filter that hides DEBUG and INFO spans."
                />
              </SimpleSelectOption>
              <SimpleSelectOption value={String(SpanLogLevel.ERROR)}>
                <FormattedMessage
                  defaultMessage="Error"
                  description="Option in the span log-level filter that hides everything below ERROR."
                />
              </SimpleSelectOption>
              <SimpleSelectOption value={String(SpanLogLevel.CRITICAL)}>
                <FormattedMessage
                  defaultMessage="Critical"
                  description="Option in the span log-level filter that shows only CRITICAL spans."
                />
              </SimpleSelectOption>
            </SimpleSelect>
          </ApplyDesignSystemContextOverrides>
          <Typography.Text color="secondary">
            <FormattedMessage
              defaultMessage="Settings"
              description="Section label for filter settings in the trace explorer."
            />
          </Typography.Text>
          <Checkbox
            componentId="codegen_no_dynamic_js_packages_web_shared_src_model_trace_explorer_timeline_tree_timelinetreefilterbutton_83"
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
            componentId="codegen_no_dynamic_js_packages_web_shared_src_model_trace_explorer_timeline_tree_timelinetreefilterbutton_111"
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
