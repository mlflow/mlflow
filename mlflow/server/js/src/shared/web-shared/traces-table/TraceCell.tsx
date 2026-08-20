import { forwardRef, memo, useCallback, useEffect, useRef, useState } from 'react';
import {
  ArrowRightIcon,
  CheckCircleIcon,
  ClockIcon,
  HoverCard,
  Tag,
  TokenIcon,
  Tooltip,
  Typography,
  useDesignSystemTheme,
  XCircleIcon,
} from '@databricks/design-system';
import type { CSSObject, Theme } from '@emotion/react';
import { defineMessages, FormattedMessage, type IntlShape, useIntl } from '@databricks/i18n';
import { CopyActionButton } from '../copy/CopyActionButton';
import { getTimeAgoStrings } from '../browse/TimeAgo';
import { formatCostUSD } from '../model-trace-explorer/CostUtils';
import {
  createTraceV4LongIdentifier,
  getTraceCost,
  getTraceTokenUsage,
} from '../model-trace-explorer/ModelTraceExplorer.utils';
import { SESSION_ID_METADATA_KEY } from '../model-trace-explorer/constants';
import type { ModelTraceInfoV3 } from '../model-trace-explorer/ModelTrace.types';
import { doesTraceSupportV4API } from '../genai-traces-table/utils/TraceLocationUtils';
import { getTraceInfoInputs, getTraceInfoOutputs } from '../genai-traces-table/utils/TraceUtils';
import { Link } from '../genai-traces-table/utils/RoutingUtils';
import { SourceCellRenderer } from '../genai-traces-table/cellRenderers/Source/SourceRenderer';
import type { SessionHrefGetter } from './types';
import { formatTraceDuration } from './formatTraceDuration';

// Module-local static analytics-id namespace. The `@databricks/no-dynamic-property-value` lint rule
// requires every `componentId` to be statically determinable, so a runtime-injected prefix isn't
// possible — an in-file const (resolved to a literal) is how these cells share a namespace.
const COMPONENT_ID = 'web-shared.traces-table';
const CELL_OVERLAY_MAX_WIDTH = 300;
// DuBois icons are 16px by default and look oversized beside the small cell text. Size them via
// `fontSize` (their SVG is 1em, so this scales the glyph and its line-box together — width/height
// alone shrinks the glyph but leaves a 16px box, dropping it below the text).
const CELL_ICON_SIZE = 13;

const WrappedTooltipText = ({ children }: { children: React.ReactNode }) => (
  <span css={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>{children}</span>
);

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

const TraceTextCell = ({ value }: { value?: string }) => {
  if (!value) {
    return <EmptyValue />;
  }
  return (
    <Tooltip
      componentId={`${COMPONENT_ID}.cell.text-tooltip`}
      content={<WrappedTooltipText>{value}</WrappedTooltipText>}
      maxWidth={CELL_OVERLAY_MAX_WIDTH}
    >
      <span css={truncateCss}>{value}</span>
    </Tooltip>
  );
};

/** User identity, matching the legacy metadata-first fallback to the older tag. */
export const TraceUserCell = ({ trace }: { trace: ModelTraceInfoV3 }): JSX.Element => (
  <TraceTextCell value={trace.trace_metadata?.['mlflow.trace.user'] || trace.tags?.['mlflow.user']} />
);

/** Display name recorded on the trace. */
export const TraceNameCell = ({ trace }: { trace: ModelTraceInfoV3 }): JSX.Element => (
  <TraceTextCell value={trace.tags?.['mlflow.traceName']} />
);

/** Notebook/job/project source, using the established legacy source renderer. */
export const TraceSourceCell = ({ trace }: { trace: ModelTraceInfoV3 }): JSX.Element => (
  <div onClick={(event) => event.stopPropagation()}>
    <SourceCellRenderer traceInfo={trace} isComparing={false} />
  </div>
);

/** Experiment-scoped run name supplied by the product consumer. */
export const TraceRunNameCell = ({
  trace,
  renderRunName,
}: {
  trace: ModelTraceInfoV3;
  renderRunName?: (trace: ModelTraceInfoV3) => React.ReactNode;
}): JSX.Element => {
  const rendered = renderRunName?.(trace);
  return rendered ? <div onClick={(event) => event.stopPropagation()}>{rendered}</div> : <EmptyValue />;
};

interface TraceCellProps {
  trace: ModelTraceInfoV3;
  /**
   * Opens the trace drawer for `trace`. Must be a stable reference so `React.memo` holds; the
   * per-trace closure is built inside the cell, not baked into the column def.
   */
  onSelect: (trace: ModelTraceInfoV3) => void;
  accessibleLabel: string;
}

interface TraceTagsCellProps extends TraceCellProps {
  /** Toggle a filter when a tag pill is clicked. Absent → pills render as plain (non-clickable) tags. */
  onFilterByTag?: (key: string, value: string) => void;
}

/** Trace id — monospace text that opens the drawer on click, plus a hover-revealed copy button. */
export const TraceIdCell: React.MemoExoticComponent<(props: TraceCellProps) => JSX.Element> = memo(
  function TraceIdCell({ trace, onSelect, accessibleLabel }: TraceCellProps) {
    const { theme } = useDesignSystemTheme();
    // CopyActionButton keeps its "Copied" tooltip open 3s, which would linger over this
    // hover-hidden button — so drive the tooltip here: open on hover/focus, brief flash on copy.
    const [copyTooltipOpen, setCopyTooltipOpen] = useState(false);
    const copiedFlashTimer = useRef<number>();
    useEffect(() => () => window.clearTimeout(copiedFlashTimer.current), []);
    const flashCopied = () => {
      setCopyTooltipOpen(true);
      window.clearTimeout(copiedFlashTimer.current);
      copiedFlashTimer.current = window.setTimeout(() => setCopyTooltipOpen(false), 1000);
    };
    return (
      <span
        css={{
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.xs,
          width: '100%',
          minWidth: 0,
          '&:hover .trace-id-copy, &:focus-within .trace-id-copy': { opacity: 1 },
        }}
      >
        <Tooltip
          componentId={`${COMPONENT_ID}.cell.trace-id-tooltip`}
          content={<WrappedTooltipText>{trace.trace_id}</WrappedTooltipText>}
          maxWidth={CELL_OVERLAY_MAX_WIDTH}
        >
          <CellActivator
            componentId={`${COMPONENT_ID}.cell.trace-id`}
            onActivate={() => onSelect(trace)}
            accessibleLabel={accessibleLabel}
            css={{ display: 'block', flex: 1, minWidth: 0 }}
          >
            <span
              css={[
                truncateCss,
                { fontFamily: 'monospace', fontSize: theme.typography.fontSizeSm, color: theme.colors.textPrimary },
              ]}
            >
              {trace.trace_id}
            </span>
          </CellActivator>
        </Tooltip>
        {/* stopPropagation so copy doesn't open the drawer. Copy the full V4 identifier (SDK
          requires the location prefix) when supported, else the bare id. */}
        <span
          className="trace-id-copy"
          onClick={(event) => event.stopPropagation()}
          css={{ flexShrink: 0, opacity: 0, transition: 'opacity 0.1s ease' }}
        >
          <CopyActionButton
            componentId={`${COMPONENT_ID}.cell.trace-id-copy`}
            copyText={doesTraceSupportV4API(trace) ? createTraceV4LongIdentifier(trace) : trace.trace_id}
            onCopy={flashCopied}
            tooltipProps={{ open: copyTooltipOpen, onOpenChange: setCopyTooltipOpen }}
          />
        </span>
      </span>
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
  const { theme } = useDesignSystemTheme();
  if (!value) {
    return <EmptyValue />;
  }
  return (
    <Tooltip
      componentId={`${componentId}-tooltip`}
      content={<WrappedTooltipText>{value}</WrappedTooltipText>}
      maxWidth={CELL_OVERLAY_MAX_WIDTH}
    >
      <CellActivator
        componentId={componentId}
        onActivate={onActivate}
        accessibleLabel={accessibleLabel}
        css={{ display: 'block', width: '100%' }}
      >
        {/* Plain text color, not link-blue: the whole row is the click target, so the preview reads as
          content rather than a link. */}
        <span css={[truncateCss, { color: theme.colors.textPrimary }]}>{value}</span>
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
 * Session id as flat truncated text (full id in a tooltip), "-" when absent. When `getSessionHref`
 * returns a `To` the text is a `Link` to the session (stopPropagation so it doesn't open the drawer;
 * Cmd/Ctrl-click still opens a new tab), with a hover-revealed arrow; otherwise plain text.
 */
export const TraceSessionCell: React.MemoExoticComponent<(props: TraceSessionCellProps) => JSX.Element> = memo(
  function TraceSessionCell({ trace, getSessionHref }: TraceSessionCellProps) {
    const { theme } = useDesignSystemTheme();
    const sessionId = trace.trace_metadata?.[SESSION_ID_METADATA_KEY];
    if (!sessionId) {
      return <EmptyValue />;
    }
    const text = <span css={[truncateCss, { color: theme.colors.textPrimary }]}>{sessionId}</span>;
    const to = getSessionHref?.({ trace, sessionId });
    if (!to) {
      return (
        <Tooltip
          componentId={`${COMPONENT_ID}.cell.session-tooltip`}
          content={<WrappedTooltipText>{sessionId}</WrappedTooltipText>}
          maxWidth={CELL_OVERLAY_MAX_WIDTH}
        >
          <span css={{ display: 'block', maxWidth: '100%' }}>{text}</span>
        </Tooltip>
      );
    }
    return (
      <Tooltip
        componentId={`${COMPONENT_ID}.cell.session-tooltip`}
        content={<WrappedTooltipText>{sessionId}</WrappedTooltipText>}
        maxWidth={CELL_OVERLAY_MAX_WIDTH}
      >
        <span
          css={{
            display: 'flex',
            alignItems: 'center',
            width: '100%',
            minWidth: 0,
            '&:hover .session-jump, &:focus-within .session-jump': { opacity: 1 },
            // The router Link renders an <a>; scope its layout here so Link only carries router props.
            '& > a': { display: 'flex', alignItems: 'center', gap: theme.spacing.xs, minWidth: 0, flex: 1 },
          }}
        >
          <Link componentId={`${COMPONENT_ID}.cell.session-link`} to={to} onClick={(event) => event.stopPropagation()}>
            {text}
            <ArrowRightIcon
              className="session-jump"
              css={{
                flexShrink: 0,
                fontSize: 12,
                color: theme.colors.textSecondary,
                opacity: 0,
                transition: 'opacity 0.1s ease',
              }}
            />
          </Link>
        </span>
      </Tooltip>
    );
  },
);

// Icon + label, no pill. STATE_UNSPECIFIED has no label — those traces render `EmptyValue`.
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

export const getTraceStateLabel = (state: ModelTraceInfoV3['state'], intl: IntlShape): string =>
  state && state !== 'STATE_UNSPECIFIED' ? intl.formatMessage(STATE_LABELS[state]) : '-';

// DuBois icons apply their semantic `color` prop with `!important`; a plain `css={{ color }}` loses to
// the icon's own default-color rule, so the color must go through the prop, not the `css`.
const stateIcon = (state: ModelTraceInfoV3['state']) => {
  if (state === 'IN_PROGRESS') {
    return <ClockIcon color="warning" css={{ fontSize: CELL_ICON_SIZE }} />;
  }
  if (state === 'OK') {
    return <CheckCircleIcon color="success" css={{ fontSize: CELL_ICON_SIZE }} />;
  }
  if (state === 'ERROR') {
    return <XCircleIcon color="danger" css={{ fontSize: CELL_ICON_SIZE }} />;
  }
  return null;
};

// The label color matches its icon so the word and icon read as one colored status.
const stateLabelColor = (state: ModelTraceInfoV3['state'], theme: Theme): string | undefined => {
  if (state === 'IN_PROGRESS') {
    return theme.colors.textValidationWarning;
  }
  if (state === 'OK') {
    return theme.colors.textValidationSuccess;
  }
  if (state === 'ERROR') {
    return theme.colors.textValidationDanger;
  }
  return undefined;
};

/**
 * Trace state as icon + label (no pill): OK → green check, ERROR → red X (with the `error_message`
 * on hover when present), IN_PROGRESS → amber clock. STATE_UNSPECIFIED renders "-".
 */
export const TraceStateCell: React.MemoExoticComponent<(props: { trace: ModelTraceInfoV3 }) => JSX.Element> = memo(
  function TraceStateCell({ trace }: { trace: ModelTraceInfoV3 }) {
    const { theme } = useDesignSystemTheme();
    const intl = useIntl();
    const state = trace.state || 'STATE_UNSPECIFIED';

    if (state === 'STATE_UNSPECIFIED') {
      return <EmptyValue />;
    }

    const badge = (
      <span css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.sm, maxWidth: '100%' }}>
        {stateIcon(state)}
        <span css={[truncateCss, { color: stateLabelColor(state, theme) }]}>
          {intl.formatMessage(STATE_LABELS[state])}
        </span>
      </span>
    );

    // Surface the trace's error message on hover, but only on the error badge and only when a
    // non-empty message is present. A focusable wrapper makes the tooltip keyboard-reachable. OSS's
    // `ModelTraceInfoV3` has no top-level `error_message` (its `SearchTracesV3` payload never sets
    // one), so read it optionally through a cast rather than the typed field.
    const errorMessage = (trace as { error_message?: string }).error_message;
    if (state === 'ERROR' && errorMessage) {
      return (
        <Tooltip
          componentId={`${COMPONENT_ID}.cell.state-error-tooltip`}
          content={<WrappedTooltipText>{errorMessage}</WrappedTooltipText>}
          maxWidth={CELL_OVERLAY_MAX_WIDTH}
        >
          <span role="button" tabIndex={0} css={{ display: 'inline-flex', alignItems: 'center', width: 'fit-content' }}>
            {badge}
          </span>
        </Tooltip>
      );
    }

    return badge;
  },
);

/** Humanized "X ago" start time with the full datetime in a hover tooltip. */
export const TraceStartTimeCell: React.MemoExoticComponent<(props: { trace: ModelTraceInfoV3 }) => JSX.Element> = memo(
  function TraceStartTimeCell({ trace }: { trace: ModelTraceInfoV3 }) {
    const { theme } = useDesignSystemTheme();
    const intl = useIntl();
    const date = trace.request_time ? new Date(trace.request_time) : undefined;
    // Guard against a missing or unparseable timestamp so a malformed row renders "-" not "Invalid Date".
    if (!date || Number.isNaN(date.getTime())) {
      return <EmptyValue />;
    }
    const { displayText, tooltipTitle } = getTimeAgoStrings({ date, intl });
    return (
      <Tooltip
        componentId={`${COMPONENT_ID}.cell.start-time-tooltip`}
        content={<WrappedTooltipText>{tooltipTitle}</WrappedTooltipText>}
        maxWidth={CELL_OVERLAY_MAX_WIDTH}
      >
        {/* Muted: the relative time is secondary to the row's content. */}
        <span css={[truncateCss, { color: theme.colors.textSecondary }]}>{displayText}</span>
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
    const { theme } = useDesignSystemTheme();
    if (!trace.execution_duration) {
      return <EmptyValue />;
    }
    const formatted = formatTraceDuration(trace.execution_duration) ?? trace.execution_duration;
    const content = (
      <span css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.sm, maxWidth: '100%' }}>
        <ClockIcon css={{ color: theme.colors.textSecondary, fontSize: CELL_ICON_SIZE }} />
        <span css={truncateCss}>{formatted}</span>
      </span>
    );
    return (
      <Tooltip
        componentId={`${COMPONENT_ID}.cell.duration-tooltip`}
        content={<WrappedTooltipText>{trace.execution_duration}</WrappedTooltipText>}
        maxWidth={CELL_OVERLAY_MAX_WIDTH}
      >
        {content}
      </Tooltip>
    );
  },
);

/** Total token count in a tag, with an input/output breakdown on hover. */
export const TraceTokensCell: React.MemoExoticComponent<(props: { trace: ModelTraceInfoV3 }) => JSX.Element> = memo(
  function TraceTokensCell({ trace }: { trace: ModelTraceInfoV3 }) {
    const { theme } = useDesignSystemTheme();
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
    const content = (
      <span css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.sm, maxWidth: '100%' }}>
        <TokenIcon css={{ color: theme.colors.textSecondary, fontSize: CELL_ICON_SIZE }} />
        <span css={truncateCss}>{usage.total_tokens}</span>
      </span>
    );
    if (parts.length === 0) {
      return content;
    }
    return (
      <Tooltip
        componentId={`${COMPONENT_ID}.cell.tokens-tooltip`}
        content={<WrappedTooltipText>{parts.join(' · ')}</WrappedTooltipText>}
        maxWidth={CELL_OVERLAY_MAX_WIDTH}
      >
        {content}
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
    const content = <span css={truncateCss}>{formatCostUSD(cost.total_cost)}</span>;
    if (parts.length === 0) {
      return content;
    }
    return (
      <Tooltip
        componentId={`${COMPONENT_ID}.cell.cost-tooltip`}
        content={<WrappedTooltipText>{parts.join(' · ')}</WrappedTooltipText>}
        maxWidth={CELL_OVERLAY_MAX_WIDTH}
      >
        {content}
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
