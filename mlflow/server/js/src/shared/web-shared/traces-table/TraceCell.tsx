import { forwardRef, memo } from 'react';
import {
  CheckCircleIcon,
  ClockIcon,
  HoverCard,
  Tag,
  type TagColors,
  Tooltip,
  Typography,
  useDesignSystemTheme,
  XCircleIcon,
} from '@databricks/design-system';
import type { CSSObject, Theme } from '@emotion/react';
import { defineMessages, FormattedMessage, type IntlShape, useIntl } from '@databricks/i18n';
import { getTimeAgoStrings } from '../browse/TimeAgo';
import { formatCostUSD } from '../model-trace-explorer/CostUtils';
import { getTraceCost, getTraceTokenUsage } from '../model-trace-explorer/ModelTraceExplorer.utils';
import { SESSION_ID_METADATA_KEY } from '../model-trace-explorer/constants';
import type { ModelTraceInfoV3 } from '../model-trace-explorer/ModelTrace.types';
import { getTraceInfoInputs, getTraceInfoOutputs } from '../genai-traces-table/utils/TraceUtils';
import { Link } from '../genai-traces-table/utils/RoutingUtils';
import type { SessionHrefGetter } from './types';
import { formatTraceDuration } from './formatTraceDuration';

// Module-local static analytics-id namespace. The `@databricks/no-dynamic-property-value` lint rule
// requires every `componentId` to be statically determinable, so a runtime-injected prefix isn't
// possible — an in-file const (resolved to a literal) is how these cells share a namespace.
const COMPONENT_ID = 'web-shared.traces-table';

/** Single-line truncation with trailing ellipsis. */
const truncateCss: CSSObject = {
  whiteSpace: 'nowrap',
  overflow: 'hidden',
  textOverflow: 'ellipsis',
  display: 'inline-block',
  maxWidth: '100%',
  verticalAlign: 'middle',
};

const ACTIVATOR_BASE_CSS: CSSObject = {
  display: 'inline-block',
  verticalAlign: 'middle',
  cursor: 'pointer',
  maxWidth: '100%',
  borderRadius: 2,
  '&:focus-visible': {
    outline: '2px solid currentColor',
    outlineOffset: 2,
  },
};

interface CellActivatorProps extends React.HTMLAttributes<HTMLSpanElement> {
  /** Opens the trace drawer. Wraps a real `role="button"` so keyboard users can activate it. */
  onActivate: () => void;
  /** Spoken label for screen readers — includes the trace id + which column. */
  accessibleLabel: string;
  componentId: string;
  children: React.ReactNode;
  css?: CSSObject;
}

/**
 * Inline activator shared by the clickable trace cells. Keyboard users tab to it and press
 * Enter/Space; mouse users click the cell. `forwardRef` + `{...rest}` is load-bearing so a
 * wrapping Radix `Tooltip.Trigger asChild` can inject its hover/focus handlers and ref.
 */
const CellActivator = forwardRef<HTMLSpanElement, CellActivatorProps>(function CellActivator(
  { onActivate, accessibleLabel, componentId, children, css: extraCss, ...rest },
  ref,
) {
  return (
    <span
      {...rest}
      ref={ref}
      role="button"
      tabIndex={0}
      aria-label={accessibleLabel}
      data-component-id={componentId}
      onClick={(event) => {
        rest.onClick?.(event);
        onActivate();
      }}
      onKeyDown={(event) => {
        rest.onKeyDown?.(event);
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          onActivate();
        }
      }}
      css={extraCss ? [ACTIVATOR_BASE_CSS, extraCss] : ACTIVATOR_BASE_CSS}
    >
      {children}
    </span>
  );
});

const EmptyValue = () => <Typography.Text color="secondary">-</Typography.Text>;

interface TraceCellProps {
  trace: ModelTraceInfoV3;
  /**
   * Opens the trace drawer for `trace`. A stable reference (the table's memoized `onTraceSelected`)
   * so the memoized cell's props don't change per render; the per-trace closure is built *inside* the
   * cell (`() => onSelect(trace)`), keeping the closure out of the parent's column def where a fresh
   * `onActivate` each render would defeat `React.memo`.
   */
  onSelect: (trace: ModelTraceInfoV3) => void;
  accessibleLabel: string;
}

interface TraceTagsCellProps extends TraceCellProps {
  /** Toggle a filter when a tag pill is clicked. Absent → pills render as plain (non-clickable) tags. */
  onFilterByTag?: (key: string, value: string) => void;
}

/** Trace id — plain monospace text (not a link), opens the drawer, full id in a tooltip. */
export const TraceIdCell: React.MemoExoticComponent<(props: TraceCellProps) => JSX.Element> = memo(
  function TraceIdCell({ trace, onSelect, accessibleLabel }: TraceCellProps) {
    const { theme } = useDesignSystemTheme();
    return (
      <Tooltip componentId={`${COMPONENT_ID}.cell.trace-id-tooltip`} content={trace.trace_id}>
        <CellActivator
          componentId={`${COMPONENT_ID}.cell.trace-id`}
          onActivate={() => onSelect(trace)}
          accessibleLabel={accessibleLabel}
          css={{ display: 'block', width: '100%' }}
        >
          <span css={[truncateCss, { fontFamily: 'monospace', fontSize: theme.typography.fontSizeSm }]}>
            {trace.trace_id}
          </span>
        </CellActivator>
      </Tooltip>
    );
  },
);

/** One-line text preview cell (input / output). Full value on hover; opens the drawer on click. */
const TracePreviewCell = ({
  value,
  onActivate,
  accessibleLabel,
  componentId,
}: {
  value: string;
  onActivate: () => void;
  accessibleLabel: string;
  componentId: string;
}) => {
  if (!value) {
    return <EmptyValue />;
  }
  return (
    <Tooltip
      componentId={`${componentId}-tooltip`}
      content={<span css={{ maxWidth: 480, whiteSpace: 'pre-wrap' }}>{value}</span>}
    >
      <CellActivator
        componentId={componentId}
        onActivate={onActivate}
        accessibleLabel={accessibleLabel}
        css={{ display: 'block', width: '100%' }}
      >
        <span css={truncateCss}>{value}</span>
      </CellActivator>
    </Tooltip>
  );
};

export const TraceInputCell: React.MemoExoticComponent<(props: TraceCellProps) => JSX.Element> = memo(
  function TraceInputCell({ trace, onSelect, accessibleLabel }: TraceCellProps) {
    return (
      <TracePreviewCell
        value={getTraceInfoInputs(trace)}
        onActivate={() => onSelect(trace)}
        accessibleLabel={accessibleLabel}
        componentId={`${COMPONENT_ID}.cell.input`}
      />
    );
  },
);

export const TraceOutputCell: React.MemoExoticComponent<(props: TraceCellProps) => JSX.Element> = memo(
  function TraceOutputCell({ trace, onSelect, accessibleLabel }: TraceCellProps) {
    return (
      <TracePreviewCell
        value={getTraceInfoOutputs(trace)}
        onActivate={() => onSelect(trace)}
        accessibleLabel={accessibleLabel}
        componentId={`${COMPONENT_ID}.cell.output`}
      />
    );
  },
);

interface TraceSessionCellProps {
  trace: ModelTraceInfoV3;
  /** Resolves the session link's destination, or `undefined` to render the session as plain text. */
  getSessionHref?: SessionHrefGetter;
}

/**
 * Session id from trace metadata as a Tag label — single-line truncated with the full id in the
 * tag's `title`; "-" when absent. The only product coupling is the *URL*: when `getSessionHref`
 * returns a `To`, the tag is wrapped in a `Link` (which `stopPropagation`s so clicking it navigates
 * instead of opening the trace drawer); otherwise the tag renders as plain text.
 */
export const TraceSessionCell: React.MemoExoticComponent<(props: TraceSessionCellProps) => JSX.Element> = memo(
  function TraceSessionCell({ trace, getSessionHref }: TraceSessionCellProps) {
    const sessionId = trace.trace_metadata?.[SESSION_ID_METADATA_KEY];
    if (!sessionId) {
      return <EmptyValue />;
    }
    const tag = (
      <Tag componentId={`${COMPONENT_ID}.cell.session`} css={{ width: 'fit-content', maxWidth: '100%' }}>
        <span css={truncateCss} title={sessionId}>
          {sessionId}
        </span>
      </Tag>
    );
    const to = getSessionHref?.({ trace, sessionId });
    if (!to) {
      return tag;
    }
    return (
      <Link componentId={`${COMPONENT_ID}.cell.session-link`} to={to} onClick={(event) => event.stopPropagation()}>
        {tag}
      </Link>
    );
  },
);

// State badge config — colored Tag + icon + label. STATE_UNSPECIFIED intentionally has no label;
// those traces render `EmptyValue`.
const STATE_LABELS = defineMessages({
  IN_PROGRESS: {
    defaultMessage: 'In progress',
    description: 'Traces table state badge label for an in-progress trace',
  },
  OK: {
    defaultMessage: 'OK',
    description: 'Traces table state badge label for a successful trace',
  },
  ERROR: {
    defaultMessage: 'Error',
    description: 'Traces table state badge label for an errored trace',
  },
});

const stateTagColor = (state: ModelTraceInfoV3['state']): TagColors | undefined => {
  if (state === 'IN_PROGRESS') {
    return 'lemon';
  }
  if (state === 'OK') {
    return 'teal';
  }
  if (state === 'ERROR') {
    return 'coral';
  }
  return undefined;
};

const stateIcon = (state: ModelTraceInfoV3['state'], theme: Theme) => {
  if (state === 'IN_PROGRESS') {
    return <ClockIcon css={{ color: theme.colors.textValidationWarning, width: 14, height: 14 }} />;
  }
  if (state === 'OK') {
    return <CheckCircleIcon css={{ color: theme.colors.textValidationSuccess, width: 14, height: 14 }} />;
  }
  if (state === 'ERROR') {
    return <XCircleIcon css={{ color: theme.colors.textValidationDanger, width: 14, height: 14 }} />;
  }
  return null;
};

/**
 * Trace state as a colored badge: OK → teal check, ERROR → coral X (with the `error_message` on
 * hover when present), IN_PROGRESS → lemon clock. STATE_UNSPECIFIED renders "-".
 */
export const TraceStateCell: React.MemoExoticComponent<(props: { trace: ModelTraceInfoV3 }) => JSX.Element> = memo(
  function TraceStateCell({ trace }: { trace: ModelTraceInfoV3 }) {
    const { theme } = useDesignSystemTheme();
    const intl = useIntl();
    const state = trace.state || 'STATE_UNSPECIFIED';

    if (state === 'STATE_UNSPECIFIED') {
      return <EmptyValue />;
    }

    const tag = (
      <Tag
        color={stateTagColor(state)}
        componentId={`${COMPONENT_ID}.cell.state`}
        css={{ display: 'inline-flex', alignItems: 'center', width: 'fit-content' }}
      >
        {stateIcon(state, theme)}
        <span css={{ marginLeft: theme.spacing.xs }}>{intl.formatMessage(STATE_LABELS[state])}</span>
      </Tag>
    );

    // Surface the trace's error message on hover, but only on the error badge and only when a
    // non-empty message is present. A focusable wrapper makes the tooltip keyboard-reachable (the
    // non-interactive Tag renders with tabIndex={-1}). OSS's `ModelTraceInfoV3` doesn't declare a
    // top-level `error_message` (its `SearchTracesV3` payload never sets one), so read it optionally
    // — the tooltip stays latent until a backend populates it.
    const errorMessage = (trace as { error_message?: string }).error_message;
    if (state === 'ERROR' && errorMessage) {
      return (
        <Tooltip componentId={`${COMPONENT_ID}.cell.state-error-tooltip`} content={errorMessage}>
          <span role="button" tabIndex={0} css={{ display: 'inline-flex', alignItems: 'center', width: 'fit-content' }}>
            {tag}
          </span>
        </Tooltip>
      );
    }

    return tag;
  },
);

/** Humanized "X ago" start time with the full datetime in a hover tooltip. */
export const TraceStartTimeCell: React.MemoExoticComponent<(props: { trace: ModelTraceInfoV3 }) => JSX.Element> = memo(
  function TraceStartTimeCell({ trace }: { trace: ModelTraceInfoV3 }) {
    const intl = useIntl();
    const date = trace.request_time ? new Date(trace.request_time) : undefined;
    // Guard against a missing or unparseable timestamp so a malformed row renders "-" not "Invalid Date".
    if (!date || Number.isNaN(date.getTime())) {
      return <EmptyValue />;
    }
    const { displayText, tooltipTitle } = getTimeAgoStrings({ date, intl });
    return (
      <Tooltip componentId={`${COMPONENT_ID}.cell.start-time-tooltip`} content={tooltipTitle}>
        <span css={truncateCss}>{displayText}</span>
      </Tooltip>
    );
  },
);

/**
 * Duration — the API returns a proto Duration string in seconds (e.g. "32.583s"). We reformat it to
 * a magnitude-appropriate unit ("32.6s", "1.5min") for scannability, keeping the raw string as the
 * hover title. If parsing fails we show the raw value rather than dropping information.
 */
export const TraceDurationCell: React.MemoExoticComponent<(props: { trace: ModelTraceInfoV3 }) => JSX.Element> = memo(
  function TraceDurationCell({ trace }: { trace: ModelTraceInfoV3 }) {
    if (!trace.execution_duration) {
      return <EmptyValue />;
    }
    const formatted = formatTraceDuration(trace.execution_duration) ?? trace.execution_duration;
    return (
      <Tag
        icon={<ClockIcon />}
        componentId={`${COMPONENT_ID}.cell.duration`}
        css={{ width: 'fit-content', maxWidth: '100%' }}
      >
        <span css={truncateCss} title={trace.execution_duration}>
          {formatted}
        </span>
      </Tag>
    );
  },
);

/** Total token count in a tag, with an input/output breakdown on hover. */
export const TraceTokensCell: React.MemoExoticComponent<(props: { trace: ModelTraceInfoV3 }) => JSX.Element> = memo(
  function TraceTokensCell({ trace }: { trace: ModelTraceInfoV3 }) {
    // `getTraceTokenUsage` can return undefined when the metadata JSON is unparseable — guard so a
    // malformed row renders "-" instead of throwing.
    const usage = getTraceTokenUsage(trace) ?? {};
    if (!usage.total_tokens) {
      return <EmptyValue />;
    }
    const parts = [
      usage.input_tokens !== undefined ? `Input ${usage.input_tokens}` : undefined,
      usage.output_tokens !== undefined ? `Output ${usage.output_tokens}` : undefined,
    ].filter(Boolean);
    const tag = (
      <Tag componentId={`${COMPONENT_ID}.cell.tokens`} css={{ width: 'fit-content', maxWidth: '100%' }}>
        <span css={truncateCss}>{usage.total_tokens}</span>
      </Tag>
    );
    if (parts.length === 0) {
      return tag;
    }
    return (
      <Tooltip componentId={`${COMPONENT_ID}.cell.tokens-tooltip`} content={parts.join(' · ')}>
        {tag}
      </Tooltip>
    );
  },
);

/** Total cost in USD in a tag, with an input/output breakdown on hover. */
export const TraceCostCell: React.MemoExoticComponent<(props: { trace: ModelTraceInfoV3 }) => JSX.Element> = memo(
  function TraceCostCell({ trace }: { trace: ModelTraceInfoV3 }) {
    // Guard against undefined from unparseable cost metadata (see TraceTokensCell).
    const cost = getTraceCost(trace) ?? {};
    if (cost.total_cost === undefined || cost.total_cost === null) {
      return <EmptyValue />;
    }
    const parts = [
      cost.input_cost !== undefined ? `Input ${formatCostUSD(cost.input_cost)}` : undefined,
      cost.output_cost !== undefined ? `Output ${formatCostUSD(cost.output_cost)}` : undefined,
    ].filter(Boolean);
    const tag = (
      <Tag componentId={`${COMPONENT_ID}.cell.cost`} css={{ width: 'fit-content', maxWidth: '100%' }}>
        <span css={truncateCss}>{formatCostUSD(cost.total_cost)}</span>
      </Tag>
    );
    if (parts.length === 0) {
      return tag;
    }
    return (
      <Tooltip componentId={`${COMPONENT_ID}.cell.cost-tooltip`} content={parts.join(' · ')}>
        {tag}
      </Tooltip>
    );
  },
);

// Tags prefixed `mlflow.` are machine-set metadata (e.g. `mlflow.trace.sizeBytes`), not user tags —
// hidden here to match the shared tags renderer and avoid noise.
const MLFLOW_INTERNAL_TAG_PREFIX = 'mlflow.';

// Aria label for a clickable tag pill that applies a tag filter. Static first arg keeps i18n
// extraction happy; the key/value are interpolated values.
const filterByTagLabel = (intl: IntlShape, key: string, value: string) =>
  intl.formatMessage(
    {
      defaultMessage: 'Filter by tag {key}: {value}',
      description: 'Aria label for a clickable tag pill in the traces table that filters by that tag',
    },
    { key, value },
  );

const TagPill = ({
  tagKey,
  value,
  onFilterByTag,
  intl,
}: {
  tagKey: string;
  value: string;
  onFilterByTag?: (key: string, value: string) => void;
  intl: IntlShape;
}) => {
  // Without a filter handler the pill is a plain, non-interactive tag (no role/aria-label/onClick).
  const filterProps = onFilterByTag
    ? {
        role: 'button',
        'aria-label': filterByTagLabel(intl, tagKey, value),
        onClick: (event: React.MouseEvent) => {
          // Don't let the click bubble to the row (opens the drawer) or the select cell (toggles selection).
          event.stopPropagation();
          onFilterByTag(tagKey, value);
        },
      }
    : {};
  return (
    <Tag componentId={`${COMPONENT_ID}.cell.tags-pill`} color="default" css={{ maxWidth: 200 }} {...filterProps}>
      <span css={truncateCss}>
        {tagKey}: {value}
      </span>
    </Tag>
  );
};

/**
 * User tags preview with optional click-to-filter: the first non-internal tag renders as a pill
 * (clickable when `onFilterByTag` is provided), plus a "+N" affordance that opens the drawer and
 * reveals the full list on hover — each tag in that list is itself a pill. Empty (or all-internal) → "-".
 *
 * The overflow list uses `HoverCard`, not `Tooltip`: the list is *interactive* (each tag can be a
 * filter pill you click), and `Tooltip` (a) defaults to `disableHoverableContent` so the pointer
 * can't reach its content, and (b) renders on the inverted tooltip surface where a `color="default"`
 * Tag's text disappears. `HoverCard` uses the normal popover surface.
 *
 * The pills are deliberately NOT nested inside the drawer-opening `CellActivator` (`role="button"`):
 * nesting interactive elements is invalid a11y. Instead the pills and the drawer activator are
 * siblings, and each clickable pill `stopPropagation`s so a filter click never also opens the drawer.
 */
export const TraceTagsCell: React.MemoExoticComponent<(props: TraceTagsCellProps) => JSX.Element> = memo(
  function TraceTagsCell({ trace, onSelect, accessibleLabel, onFilterByTag }: TraceTagsCellProps) {
    const { theme } = useDesignSystemTheme();
    const intl = useIntl();
    const entries = Object.entries(trace.tags ?? {}).filter(([key]) => !key.startsWith(MLFLOW_INTERNAL_TAG_PREFIX));
    if (entries.length === 0) {
      return <EmptyValue />;
    }
    const [firstKey, firstValue] = entries[0];
    const restCount = entries.length - 1;

    const firstPill = <TagPill tagKey={firstKey} value={firstValue} onFilterByTag={onFilterByTag} intl={intl} />;

    // Single tag: just the pill (no "+N", no drawer affordance needed here — the row still opens the drawer).
    if (restCount === 0) {
      return <span css={{ display: 'inline-flex', maxWidth: '100%' }}>{firstPill}</span>;
    }

    // Multi-tag: the pill sits beside a "+N" that opens the drawer and, on hover, lists every tag as its
    // own pill. Activator and pills are siblings (no interactive nesting).
    return (
      <span css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.xs, maxWidth: '100%' }}>
        {firstPill}
        <HoverCard
          align="start"
          minWidth={0}
          maxWidth={480}
          trigger={
            <CellActivator
              componentId={`${COMPONENT_ID}.cell.tags`}
              onActivate={() => onSelect(trace)}
              accessibleLabel={accessibleLabel}
            >
              <Typography.Text color="secondary" size="sm">
                <FormattedMessage
                  defaultMessage="+{count}"
                  description="Compact suffix on the traces table tags column indicating how many additional tags are hidden in the preview"
                  values={{ count: restCount }}
                />
              </Typography.Text>
            </CellActivator>
          }
          content={
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
              {entries.map(([key, value]) => (
                <TagPill key={key} tagKey={key} value={value} onFilterByTag={onFilterByTag} intl={intl} />
              ))}
            </div>
          }
        />
      </span>
    );
  },
);
