import type { ComponentType } from 'react';

import { z } from 'zod';
import { createComponentImplementation } from '@a2ui/react/v0_9';
import { type ComponentApi, DynamicStringSchema } from '@a2ui/web_core/v0_9';
import {
  ChecklistIcon,
  ClockIcon,
  HashIcon,
  ListBorderIcon,
  Typography,
  useDesignSystemTheme,
  WrenchIcon,
} from '@databricks/design-system';

const ICON_NAMES = ['clock', 'list', 'wrench', 'hash', 'checklist'] as const;
type IconName = (typeof ICON_NAMES)[number];

const ICON_BY_NAME: Record<IconName, ComponentType> = {
  clock: ClockIcon,
  list: ListBorderIcon,
  wrench: WrenchIcon,
  hash: HashIcon,
  checklist: ChecklistIcon,
};

const LABEL_COLUMN_WIDTH = 200;
const ROW_HEIGHT = 28;
const INDENT_PER_DEPTH = 16;
// Reserve space on the right of the time domain so duration labels at the end of
// long bars don't overflow the chart (mirrors the gantt's "overshoot" behavior).
const DOMAIN_RIGHT_PADDING_RATIO = 0.18;

const formatMs = (ms: number): string => {
  if (ms >= 60_000) {
    return `${(ms / 60_000).toFixed(2)}min`;
  }
  if (ms >= 1000) {
    return `${(ms / 1000).toFixed(2)}s`;
  }
  return `${Math.round(ms)}ms`;
};

// Generates up to `maxTicks` "nice" axis values (1/2/5 * 10^n) spanning [0, max].
const getNiceTicks = (max: number, maxTicks = 5): number[] => {
  if (max <= 0) {
    return [0];
  }
  const rawInterval = max / maxTicks;
  const magnitude = Math.pow(10, Math.floor(Math.log10(rawInterval)));
  const residual = rawInterval / magnitude;
  const niceFraction = residual <= 1 ? 1 : residual <= 2 ? 2 : residual <= 5 ? 5 : 10;
  const interval = niceFraction * magnitude;

  const ticks: number[] = [];
  for (let tick = 0; tick <= max; tick += interval) {
    ticks.push(Number(tick.toFixed(10)));
  }
  return ticks;
};

/**
 * Schema (API) for the generic TimelineChart component. It renders a Gantt-style
 * timeline of horizontal bars, one per row, positioned by `start`/`end` offsets
 * (in milliseconds, relative to a common origin). It is domain-agnostic: callers
 * supply rows with a label, a time range, and optional indentation/color, and the
 * component computes the axis, scaling, and duration labels. Works for any number
 * of rows, mirroring the "Details & Timeline" gantt view.
 */
export const TimelineChartApi = {
  name: 'TimelineChart',
  schema: z
    .object({
      title: DynamicStringSchema.describe('Optional heading shown above the chart.').optional(),
      icon: z.enum(ICON_NAMES).describe('Optional icon shown next to the title.').optional(),
      rows: z
        .array(
          z.object({
            label: DynamicStringSchema.describe('The row label (e.g. the span name).'),
            start: z.number().describe('Start offset in milliseconds.'),
            end: z.number().describe('End offset in milliseconds.'),
            depth: z.number().describe('Indentation level for hierarchy.').default(0).optional(),
            color: z.string().describe('Optional CSS color for the bar.').optional(),
          }),
        )
        .describe('The timeline rows, in display order.'),
      emptyMessage: DynamicStringSchema.describe('Text shown when there are no rows.').optional(),
    })
    .strict(),
} satisfies ComponentApi;

const asString = (value: unknown): string => (typeof value === 'string' ? value : String(value ?? ''));
const asNumber = (value: unknown): number => (typeof value === 'number' && Number.isFinite(value) ? value : 0);

export const TimelineChart = createComponentImplementation(TimelineChartApi, ({ props }) => {
  const { theme } = useDesignSystemTheme();

  const rows = Array.isArray(props.rows) ? props.rows : [];
  const title = props.title ? asString(props.title) : undefined;
  const emptyMessage = props.emptyMessage ? asString(props.emptyMessage) : 'No spans to display.';
  const IconComponent = props.icon ? ICON_BY_NAME[props.icon as IconName] : undefined;

  const domainEnd = rows.length > 0 ? Math.max(...rows.map((row) => asNumber(row?.end))) : 0;
  const ticks = getNiceTicks(domainEnd);
  // Scale positions against a padded domain so end-of-bar duration labels fit.
  const scaleEnd = domainEnd > 0 ? domainEnd * (1 + DOMAIN_RIGHT_PADDING_RATIO) : 1;
  const toPercent = (value: number) => `${(asNumber(value) / scaleEnd) * 100}%`;

  const defaultBarColor = theme.colors.actionDefaultBackgroundPress;

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
      {title && (
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
          {IconComponent && (
            <span css={{ display: 'flex', color: theme.colors.textSecondary }}>
              <IconComponent />
            </span>
          )}
          <Typography.Text bold size="lg">
            {title}
          </Typography.Text>
        </div>
      )}

      {rows.length === 0 ? (
        <Typography.Text color="secondary">{emptyMessage}</Typography.Text>
      ) : (
        <div css={{ display: 'flex', flexDirection: 'column' }}>
          {/* Axis with tick labels, aligned to the track (offset by the label column). */}
          <div css={{ display: 'flex', height: theme.typography.lineHeightBase, flexShrink: 0 }}>
            <div css={{ width: LABEL_COLUMN_WIDTH, flexShrink: 0 }} />
            <div css={{ position: 'relative', flex: 1 }}>
              {ticks.map((tick) => (
                <Typography.Text
                  key={tick}
                  color="secondary"
                  size="sm"
                  css={{ position: 'absolute', left: toPercent(tick), whiteSpace: 'nowrap' }}
                >
                  {formatMs(tick)}
                </Typography.Text>
              ))}
            </div>
          </div>

          {/* Rows: label column + track with gridlines and a positioned bar. */}
          <div css={{ display: 'flex', flexDirection: 'column' }}>
            {rows.map((row, rowIndex) => {
              const start = asNumber(row?.start);
              const end = Math.max(asNumber(row?.end), start);
              const depth = asNumber(row?.depth);
              const barColor = row?.color ?? defaultBarColor;

              return (
                <div key={rowIndex} css={{ display: 'flex', alignItems: 'center', height: ROW_HEIGHT }}>
                  <div
                    css={{
                      width: LABEL_COLUMN_WIDTH,
                      flexShrink: 0,
                      paddingLeft: depth * INDENT_PER_DEPTH,
                      paddingRight: theme.spacing.sm,
                      display: 'flex',
                      alignItems: 'center',
                      gap: theme.spacing.xs,
                      minWidth: 0,
                    }}
                  >
                    {row?.color && (
                      <span
                        css={{ width: 8, height: 8, borderRadius: '50%', backgroundColor: row.color, flexShrink: 0 }}
                      />
                    )}
                    <Typography.Text
                      css={{
                        fontFamily: 'monospace',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {asString(row?.label)}
                    </Typography.Text>
                  </div>

                  <div css={{ position: 'relative', flex: 1, height: '100%' }}>
                    {/* gridlines */}
                    {ticks.map((tick) => (
                      <div
                        key={tick}
                        css={{
                          position: 'absolute',
                          top: 0,
                          bottom: 0,
                          left: toPercent(tick),
                          borderRight: `1px solid ${theme.colors.borderDecorative}`,
                        }}
                      />
                    ))}
                    {/* bar */}
                    <div
                      css={{
                        position: 'absolute',
                        top: '50%',
                        transform: 'translateY(-50%)',
                        left: toPercent(start),
                        width: `max(${toPercent(end - start)}, 2px)`,
                        height: 16,
                        backgroundColor: barColor,
                        borderRadius: theme.borders.borderRadiusSm,
                      }}
                    />
                    {/* duration label, just past the end of the bar */}
                    <Typography.Text
                      color="secondary"
                      size="sm"
                      css={{
                        position: 'absolute',
                        top: '50%',
                        transform: 'translateY(-50%)',
                        left: `calc(${toPercent(end)} + ${theme.spacing.xs}px)`,
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {formatMs(end - start)}
                    </Typography.Text>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
});
