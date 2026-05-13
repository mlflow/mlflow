import {
  Button,
  Checkbox,
  FilterIcon,
  InfoTooltip,
  Popover,
  Slider,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import type { SpanFilterState } from '../ModelTrace.types';
import { SpanLogLevel } from '../ModelTrace.types';
import { getDisplayNameForSpanType, getIconTypeForSpan } from '../ModelTraceExplorer.utils';
import { ModelTraceExplorerIcon } from '../ModelTraceExplorerIcon';

// Slider tracks indices 0..4; map to/from the SpanLogLevel enum (10/20/30/40/50).
const LOG_LEVEL_ORDER: readonly SpanLogLevel[] = [
  SpanLogLevel.DEBUG,
  SpanLogLevel.INFO,
  SpanLogLevel.WARNING,
  SpanLogLevel.ERROR,
  SpanLogLevel.CRITICAL,
];

const indexFromLogLevel = (level: SpanLogLevel): number => {
  const idx = LOG_LEVEL_ORDER.indexOf(level);
  return idx >= 0 ? idx : 0;
};

// Helper text shown under the slider for each level. Frames the threshold in
// terms of the *filtering effect* (what the user will see after picking it),
// not the type-to-level mapping — that mapping lives in the docs and the
// `defaultLogLevelForSpanType` helper.
const LEVEL_DESCRIPTIONS: Record<SpanLogLevel, { label: string; behavior: string }> = {
  [SpanLogLevel.DEBUG]: {
    label: 'Debug',
    behavior: 'Show all spans regardless.',
  },
  [SpanLogLevel.INFO]: {
    label: 'Info',
    behavior: 'Filter to important span types such as LLM, Tool, Retriever, Agent, and Embedding.',
  },
  [SpanLogLevel.WARNING]: {
    label: 'Warning',
    behavior: 'Filter to spans with an explicit warning level or above.',
  },
  [SpanLogLevel.ERROR]: {
    label: 'Error',
    behavior: 'Filter to spans with an exception event.',
  },
  [SpanLogLevel.CRITICAL]: {
    label: 'Critical',
    behavior: 'Filter to spans with an explicit critical level.',
  },
};

export const TimelineTreeFilterButton = ({
  spanFilterState,
  setSpanFilterState,
}: {
  spanFilterState: SpanFilterState;
  setSpanFilterState: (state: SpanFilterState) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const sliderIndex = indexFromLogLevel(spanFilterState.minLogLevel);
  const currentLevel = LOG_LEVEL_ORDER[sliderIndex];
  const description = LEVEL_DESCRIPTIONS[currentLevel];

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
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.sm,
            paddingBottom: theme.spacing.xs,
            // Pin the popover width so per-level helper text under the slider
            // (which varies in length) doesn't make the popover and slider
            // visibly resize as the user drags.
            width: 280,
          }}
        >
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
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
            <Typography.Text color="secondary">
              <FormattedMessage
                defaultMessage="Minimum log level"
                description="Section label for the minimum span log level filter in the trace explorer."
              />
            </Typography.Text>
            <InfoTooltip
              componentId="shared.model-trace-explorer.log-level-tooltip"
              content={
                <FormattedMessage
                  defaultMessage="Hide low-severity spans. Each span gets a level based on its type. Spans with an exception event are promoted to Error. Spans recorded before MLflow 3.13 don't carry a level and are treated as Debug. <link>Learn more</link>."
                  description="Tooltip explaining the minimum log level filter, including the autolog default mapping, exception-bump rule, and pre-3.13 backwards-compat note."
                  values={{
                    link: (chunks: any) => (
                      <a
                        href="https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/logging"
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        {chunks}
                      </a>
                    ),
                  }}
                />
              }
            />
          </div>
          <div
            aria-label="Minimum log level"
            css={{
              display: 'flex',
              flexDirection: 'column',
              gap: theme.spacing.xs,
              paddingTop: theme.spacing.xs,
              paddingLeft: theme.spacing.sm,
              paddingRight: theme.spacing.sm,
            }}
          >
            <Slider.Root
              min={0}
              max={LOG_LEVEL_ORDER.length - 1}
              step={1}
              value={[sliderIndex]}
              onValueChange={([nextIndex]) =>
                setSpanFilterState({
                  ...spanFilterState,
                  minLogLevel: LOG_LEVEL_ORDER[nextIndex],
                })
              }
              // Override the design-system's default 200px horizontal width via
              // inline style (rather than css), so we don't clobber the sibling
              // `display: flex` / `alignItems: center` defaults that
              // `Slider.Track` relies on to expand. A `css={{width:'100%'}}` here
              // would replace `getRootStyles()` entirely and collapse the track,
              // making the thumb unable to reach the rightmost stop.
              style={{ width: '100%' }}
            >
              <Slider.Track>
                <Slider.Range />
              </Slider.Track>
              <Slider.Thumb aria-label="Minimum log level" />
            </Slider.Root>
            <div
              css={{
                display: 'flex',
                justifyContent: 'space-between',
                fontSize: theme.typography.fontSizeSm,
                color: theme.colors.textSecondary,
              }}
            >
              {LOG_LEVEL_ORDER.map((level) => (
                <span key={level} css={{ opacity: level === currentLevel ? 1 : 0.7 }}>
                  {LEVEL_DESCRIPTIONS[level].label}
                </span>
              ))}
            </div>
            <Typography.Text size="sm" color="secondary">
              <FormattedMessage
                defaultMessage="{label}: {behavior}"
                description="Helper text under the log-level slider explaining how the currently selected level filters the trace view."
                values={{
                  label: <strong>{description.label}</strong>,
                  behavior: description.behavior,
                }}
              />
            </Typography.Text>
          </div>
        </div>
      </Popover.Content>
    </Popover.Root>
  );
};
