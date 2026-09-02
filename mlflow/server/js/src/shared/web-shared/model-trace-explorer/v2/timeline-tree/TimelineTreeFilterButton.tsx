import {
  ApplyDesignSystemContextOverrides,
  Button,
  DropdownMenu,
  FilterIcon,
  InfoTooltip,
  ListIcon,
  RefreshIcon,
  SlidersIcon,
  Tooltip,
  useSnackBar,
} from '@databricks/design-system';
import { useIntl } from '@databricks/i18n';

import type { SpanFilterState } from '../ModelTrace.types';
import { type TimelineTreeMetric, useModelTraceExplorerPreferences } from '../ModelTraceExplorerPreferencesContext';
import { getDisplayNameForSpanType } from '../ModelTraceExplorer.utils';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';

const DISPLAY_METRIC_OPTIONS = ['duration', 'tokens', 'cost'] as const satisfies readonly TimelineTreeMetric[];

// Dropdown menus render at 10000, so their contextual help must sit above the open menu.
const FILTER_OPTION_TOOLTIP_Z_INDEX = 10001;

export const TimelineTreeFilterButton = ({
  spanFilterState,
  setSpanFilterState,
  showGraph,
  onToggleGraph,
}: {
  spanFilterState: SpanFilterState;
  setSpanFilterState: (state: SpanFilterState) => void;
  showGraph?: boolean;
  onToggleGraph?: () => void;
}): JSX.Element => {
  const intl = useIntl();
  const { addSnack } = useSnackBar();
  const { timelineTreeMetrics, setTimelineTreeMetrics } = useModelTraceExplorerPreferences();
  const { refreshTrace, isRefreshingTrace } = useModelTraceExplorerViewState();

  const metricLabel = (metric: (typeof DISPLAY_METRIC_OPTIONS)[number]): string => {
    switch (metric) {
      case 'duration':
        return intl.formatMessage({ defaultMessage: 'Duration', description: 'Trace span metric option' });
      case 'tokens':
        return intl.formatMessage({ defaultMessage: 'Total tokens', description: 'Trace span metric option' });
      case 'cost':
        return intl.formatMessage({ defaultMessage: 'Estimated LLM cost', description: 'Trace span metric option' });
    }
  };

  const settingsLabel = intl.formatMessage({
    defaultMessage: 'Trace display settings',
    description: 'Accessible label for trace display settings button',
  });
  const showParentSpansLabel = intl.formatMessage({
    defaultMessage: 'Show all parent spans',
    description: 'Trace span filter option',
  });
  const showExceptionSpansLabel = intl.formatMessage({
    defaultMessage: 'Show exceptions',
    description: 'Trace span filter option',
  });
  const selectedMetricSet = new Set(timelineTreeMetrics);

  return (
    <ApplyDesignSystemContextOverrides getPopupContainer={() => document.body}>
      <DropdownMenu.Root>
        <Tooltip componentId="shared.model-trace-explorer.settings-tooltip" content={settingsLabel}>
          <DropdownMenu.Trigger asChild aria-label={settingsLabel}>
            <Button
              componentId="shared.model-trace-explorer.settings-button"
              icon={<SlidersIcon />}
              size="small"
              aria-label={settingsLabel}
            />
          </DropdownMenu.Trigger>
        </Tooltip>
        <DropdownMenu.Content align="end">
          <DropdownMenu.Sub>
            <DropdownMenu.SubTrigger>
              <DropdownMenu.IconWrapper>
                <ListIcon />
              </DropdownMenu.IconWrapper>
              {intl.formatMessage({
                defaultMessage: 'Display metric types',
                description: 'Trace settings submenu for visible span metrics',
              })}
            </DropdownMenu.SubTrigger>
            <DropdownMenu.SubContent>
              {DISPLAY_METRIC_OPTIONS.map((metric) => (
                <DropdownMenu.CheckboxItem
                  key={metric}
                  componentId="shared.model-trace-explorer.toggle-metric"
                  checked={timelineTreeMetrics.includes(metric)}
                  onSelect={(event) => event.preventDefault()}
                  onCheckedChange={(checked) => {
                    setTimelineTreeMetrics(
                      DISPLAY_METRIC_OPTIONS.filter((option) =>
                        option === metric ? checked : selectedMetricSet.has(option),
                      ),
                    );
                  }}
                >
                  <DropdownMenu.ItemIndicator />
                  {metricLabel(metric)}
                </DropdownMenu.CheckboxItem>
              ))}
            </DropdownMenu.SubContent>
          </DropdownMenu.Sub>
          <DropdownMenu.Sub>
            <DropdownMenu.SubTrigger>
              <DropdownMenu.IconWrapper>
                <FilterIcon />
              </DropdownMenu.IconWrapper>
              {intl.formatMessage({
                defaultMessage: 'Filter by span type',
                description: 'Trace settings submenu for span type filters',
              })}
            </DropdownMenu.SubTrigger>
            <DropdownMenu.SubContent>
              {Object.entries(spanFilterState.spanTypeDisplayState).map(([spanType, shouldDisplay]) => (
                <DropdownMenu.CheckboxItem
                  key={spanType}
                  componentId="shared.model-trace-explorer.toggle-span-filter"
                  checked={shouldDisplay}
                  onSelect={(event) => event.preventDefault()}
                  onCheckedChange={(checked) =>
                    setSpanFilterState({
                      ...spanFilterState,
                      spanTypeDisplayState: { ...spanFilterState.spanTypeDisplayState, [spanType]: checked },
                    })
                  }
                >
                  <DropdownMenu.ItemIndicator />
                  {getDisplayNameForSpanType(spanType)}
                </DropdownMenu.CheckboxItem>
              ))}
              <DropdownMenu.Separator />
              <DropdownMenu.CheckboxItem
                componentId="shared.model-trace-explorer.show-parent-spans"
                aria-label={showParentSpansLabel}
                checked={spanFilterState.showParents}
                onSelect={(event) => event.preventDefault()}
                onCheckedChange={(checked) => setSpanFilterState({ ...spanFilterState, showParents: checked })}
              >
                <DropdownMenu.ItemIndicator />
                {showParentSpansLabel}
                <DropdownMenu.HintColumn>
                  <InfoTooltip
                    componentId="shared.model-trace-explorer.show-parent-spans-tooltip"
                    iconTitle={intl.formatMessage({
                      defaultMessage: 'More information about showing parent spans',
                      description: 'Accessible label for help about the trace parent span filter option',
                    })}
                    content={intl.formatMessage({
                      defaultMessage: 'Always show parents of matched spans, regardless of filter conditions',
                      description: 'Tooltip explaining the trace parent span filter option',
                    })}
                    zIndex={FILTER_OPTION_TOOLTIP_Z_INDEX}
                  />
                </DropdownMenu.HintColumn>
              </DropdownMenu.CheckboxItem>
              <DropdownMenu.CheckboxItem
                componentId="shared.model-trace-explorer.show-exception-spans"
                aria-label={showExceptionSpansLabel}
                checked={spanFilterState.showExceptions}
                onSelect={(event) => event.preventDefault()}
                onCheckedChange={(checked) => setSpanFilterState({ ...spanFilterState, showExceptions: checked })}
              >
                <DropdownMenu.ItemIndicator />
                {showExceptionSpansLabel}
                <DropdownMenu.HintColumn>
                  <InfoTooltip
                    componentId="shared.model-trace-explorer.show-exception-spans-tooltip"
                    iconTitle={intl.formatMessage({
                      defaultMessage: 'More information about showing exception spans',
                      description: 'Accessible label for help about the trace exception span filter option',
                    })}
                    content={intl.formatMessage({
                      defaultMessage: 'Always show spans with exceptions, regardless of filter conditions',
                      description: 'Tooltip explaining the trace exception span filter option',
                    })}
                    zIndex={FILTER_OPTION_TOOLTIP_Z_INDEX}
                  />
                </DropdownMenu.HintColumn>
              </DropdownMenu.CheckboxItem>
            </DropdownMenu.SubContent>
          </DropdownMenu.Sub>
          {onToggleGraph && (
            <>
              <DropdownMenu.Separator />
              <DropdownMenu.CheckboxItem
                componentId="shared.model-trace-explorer.toggle-graph-button"
                checked={showGraph}
                onSelect={(event) => event.preventDefault()}
                onCheckedChange={onToggleGraph}
              >
                <DropdownMenu.ItemIndicator />
                {intl.formatMessage({
                  defaultMessage: 'Show graph',
                  description: 'Trace display setting that toggles the workflow graph',
                })}
              </DropdownMenu.CheckboxItem>
            </>
          )}
          {refreshTrace && (
            <>
              <DropdownMenu.Separator />
              <DropdownMenu.Item
                componentId="shared.model-trace-explorer.refresh-trace"
                disabled={isRefreshingTrace}
                onSelect={(event) => {
                  event.preventDefault();
                  void refreshTrace().catch(() => {
                    addSnack({
                      componentId: 'shared.model-trace-explorer.refresh-trace-error',
                      content: intl.formatMessage({
                        defaultMessage: 'Could not refresh trace',
                        description: 'Error notification when refreshing trace details fails',
                      }),
                    });
                  });
                }}
              >
                <DropdownMenu.IconWrapper>
                  <RefreshIcon spin={isRefreshingTrace} />
                </DropdownMenu.IconWrapper>
                {isRefreshingTrace
                  ? intl.formatMessage({
                      defaultMessage: 'Refreshing…',
                      description: 'Trace refresh action in progress',
                    })
                  : intl.formatMessage({
                      defaultMessage: 'Refresh trace',
                      description: 'Action to refresh trace details',
                    })}
              </DropdownMenu.Item>
            </>
          )}
        </DropdownMenu.Content>
      </DropdownMenu.Root>
    </ApplyDesignSystemContextOverrides>
  );
};
