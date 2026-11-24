import { Tag, Tooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';
import type { CSSObject, Theme } from '@emotion/react';
import { forwardRef, Fragment, useMemo } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { getTimeAgoStrings } from '@databricks/web-shared/browse';
import type { DatasetRecord } from '../hooks/useDatasetsQueries';

type RecordJsonValue = DatasetRecord['inputs'];
type RecordTags = DatasetRecord['tags'];

// Structured truncation caps for the hover preview. The tooltip is an aggressive
// *preview* — users always open the side panel for the full payload. These caps keep
// the tooltip small enough that it can't compete with the side panel as a reading
// surface, while still showing the JSON's shape (first couple of keys / items, every
// long string clipped). Truncation is applied per-node so the resulting structure
// remains valid JSON (vs. a raw `slice()` that cuts mid-token).
const TOOLTIP_MAX_STRING_LENGTH = 30;
const TOOLTIP_MAX_ARRAY_ITEMS = 2;
const TOOLTIP_MAX_OBJECT_KEYS = 2;
const TOOLTIP_MAX_DEPTH = 6;
const TOOLTIP_MAX_WIDTH = 480;

// Per-object truncation marker. Stored in a WeakMap so it (a) stays invisible to
// `Object.entries` / `Object.keys` / `Array#map` — letting the renderer's real-entry
// loops run unchanged — and (b) is dropped from `JSON.stringify` output, so the
// "result is always valid JSON" invariant in the unit tests still holds without
// special-casing the marker. `WeakKey` is the TS-lib type that exactly captures
// "anything WeakMap accepts as a key" (arrays + plain objects, what we pass), so
// the algorithm typechecks without any `as` casts.
const truncationComments = new WeakMap<WeakKey, string>();

const withTruncationComment = <T extends WeakKey>(value: T, text: string): T => {
  truncationComments.set(value, text);
  return value;
};

// Exported for unit tests; the rendered tooltip tree is covered by manual browser
// smoke.
export const getTruncationComment = (value: unknown): string | undefined => {
  if (value === null || typeof value !== 'object') return undefined;
  return truncationComments.get(value);
};

// Exported for unit tests. Radix Tooltip's portal doesn't reliably open in jsdom on
// `user.hover`, so the truncation algorithm is unit-tested here directly; the rendered
// tooltip tree is covered by the manual browser smoke test in the plan.
export const truncateJsonForTooltip = (value: unknown, depth = 0): unknown => {
  if (depth >= TOOLTIP_MAX_DEPTH) {
    if (Array.isArray(value)) {
      return value.length === 0 ? [] : withTruncationComment([], `+${value.length} more`);
    }
    if (value !== null && typeof value === 'object') {
      const keyCount = Object.keys(value).length;
      return keyCount === 0 ? {} : withTruncationComment({}, `+${keyCount} more keys`);
    }
  }
  if (typeof value === 'string') {
    return value.length > TOOLTIP_MAX_STRING_LENGTH ? `${value.slice(0, TOOLTIP_MAX_STRING_LENGTH)}…` : value;
  }
  if (Array.isArray(value)) {
    const head: unknown[] = value
      .slice(0, TOOLTIP_MAX_ARRAY_ITEMS)
      .map((item) => truncateJsonForTooltip(item, depth + 1));
    if (value.length > TOOLTIP_MAX_ARRAY_ITEMS) {
      return withTruncationComment(head, `+${value.length - TOOLTIP_MAX_ARRAY_ITEMS} more`);
    }
    return head;
  }
  if (value !== null && typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>);
    const out: Record<string, unknown> = {};
    for (const [key, v] of entries.slice(0, TOOLTIP_MAX_OBJECT_KEYS)) {
      out[key] = truncateJsonForTooltip(v, depth + 1);
    }
    if (entries.length > TOOLTIP_MAX_OBJECT_KEYS) {
      return withTruncationComment(out, `+${entries.length - TOOLTIP_MAX_OBJECT_KEYS} more keys`);
    }
    return out;
  }
  return value;
};

type JsonColors = {
  key: string;
  string: string;
  number: string;
  boolean: string;
  null: string;
  punctuation: string;
};

// The literal hexes intentionally mirror the Prism-matched palette in
// `js/packages/web-shared/src/model-trace-explorer/CollapsibleJsonViewer.tsx`
// (the canonical JSON-syntax surface in this app) so the eval-records hover preview
// reads consistently with the trace-explorer JSON view.
/* eslint-disable @databricks/no-hardcoded-colors */
const getJsonColors = (isDarkMode: boolean, textPrimary: string, textSecondary: string): JsonColors => ({
  key: isDarkMode ? '#5DFAFC' : '#39adb5',
  string: isDarkMode ? '#ffffff' : textPrimary,
  number: isDarkMode ? '#3AACE2' : '#f5871f',
  boolean: isDarkMode ? '#ffffff' : textPrimary,
  null: isDarkMode ? '#ffffff' : textPrimary,
  punctuation: textSecondary,
});
/* eslint-enable @databricks/no-hardcoded-colors */

const TOOLTIP_JSON_INDENT = '  ';

const renderJsonNode = (value: unknown, depth: number, colors: JsonColors): React.ReactNode => {
  if (value === null) return <span css={{ color: colors.null }}>null</span>;
  if (typeof value === 'string') return <span css={{ color: colors.string }}>"{value}"</span>;
  if (typeof value === 'number') return <span css={{ color: colors.number }}>{String(value)}</span>;
  if (typeof value === 'boolean') return <span css={{ color: colors.boolean }}>{String(value)}</span>;
  if (Array.isArray(value)) {
    const truncationText = getTruncationComment(value);
    if (value.length === 0 && !truncationText) return <span css={{ color: colors.punctuation }}>[]</span>;
    const innerIndent = TOOLTIP_JSON_INDENT.repeat(depth + 1);
    const closeIndent = TOOLTIP_JSON_INDENT.repeat(depth);
    return (
      <>
        <span css={{ color: colors.punctuation }}>[</span>
        {value.map((item, i) => {
          const hasFollowing = i < value.length - 1 || Boolean(truncationText);
          return (
            <Fragment key={i}>
              {'\n' + innerIndent}
              {renderJsonNode(item, depth + 1, colors)}
              {hasFollowing ? <span css={{ color: colors.punctuation }}>,</span> : null}
            </Fragment>
          );
        })}
        {truncationText && (
          <>
            {'\n' + innerIndent}
            <span css={{ color: colors.punctuation, fontStyle: 'italic' }}>{truncationText}</span>
          </>
        )}
        {'\n' + closeIndent}
        <span css={{ color: colors.punctuation }}>]</span>
      </>
    );
  }
  if (typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>);
    const truncationText = getTruncationComment(value);
    if (entries.length === 0 && !truncationText) return <span css={{ color: colors.punctuation }}>{'{}'}</span>;
    const innerIndent = TOOLTIP_JSON_INDENT.repeat(depth + 1);
    const closeIndent = TOOLTIP_JSON_INDENT.repeat(depth);
    return (
      <>
        <span css={{ color: colors.punctuation }}>{'{'}</span>
        {entries.map(([key, val], i) => {
          const hasFollowing = i < entries.length - 1 || Boolean(truncationText);
          return (
            <Fragment key={key}>
              {'\n' + innerIndent}
              <span css={{ color: colors.key }}>"{key}"</span>
              <span css={{ color: colors.punctuation }}>: </span>
              {renderJsonNode(val, depth + 1, colors)}
              {hasFollowing ? <span css={{ color: colors.punctuation }}>,</span> : null}
            </Fragment>
          );
        })}
        {truncationText && (
          <>
            {'\n' + innerIndent}
            <span css={{ color: colors.punctuation, fontStyle: 'italic' }}>{truncationText}</span>
          </>
        )}
        {'\n' + closeIndent}
        <span css={{ color: colors.punctuation }}>{'}'}</span>
      </>
    );
  }
  return <span css={{ color: colors.null }}>{String(value)}</span>;
};

/** Single-line truncation with a trailing ellipsis. Used by every cell that risks overflowing
 * its column. */
export const truncateCss: CSSObject = {
  whiteSpace: 'nowrap',
  overflow: 'hidden',
  textOverflow: 'ellipsis',
  display: 'inline-block',
  maxWidth: '100%',
  verticalAlign: 'middle',
};

const monospaceCss = (theme: Theme): CSSObject => ({
  ...truncateCss,
  fontFamily: 'monospace',
  fontSize: theme.typography.fontSizeSm,
});

// Shared "tabbable inline activator" styles: a span the user can Tab to and activate with
// Enter/Space. Stripped of default button visuals; the cell visuals come from inner content
// (a monospace span, a link, a tag pill, etc., depending on the cell).
const ACTIVATOR_BASE_CSS: CSSObject = {
  display: 'inline-block',
  verticalAlign: 'middle',
  cursor: 'pointer',
  // Make sure :focus-visible draws a visible ring against the row hover background.
  borderRadius: 2,
  '&:focus-visible': {
    outline: '2px solid currentColor',
    outlineOffset: 2,
  },
};

interface CellActivatorProps extends React.HTMLAttributes<HTMLSpanElement> {
  /** Drawer-open callback. Wraps a real `role="button"` so keyboard users can activate. */
  onActivate: () => void;
  /** Spoken label for screen readers. Includes the record id + which column. */
  accessibleLabel: string;
  /** Component id forwarded to analytics; differs per column. */
  componentId: string;
  children: React.ReactNode;
  /** Extra css overlaid on the activator span (max-width, font sizing, etc.). */
  css?: CSSObject;
}

/**
 * Single-source inline activator used by RecordIdCell, JsonPreviewCell, and TagsPreviewCell.
 * Keyboard users tab to this span and press Enter/Space to open the row's drawer; mouse
 * users click the cell as usual.
 *
 * `forwardRef` + `{...rest}` is load-bearing: the surrounding Du Bois `Tooltip` is Radix-based
 * and renders its trigger as `<Tooltip.Trigger asChild>`, which uses Radix `Slot` to clone
 * this child and inject `onMouseEnter`/`onMouseLeave`/`onPointerEnter`/`onPointerLeave`/
 * `onFocus`/`onBlur` + a composed ref. Dropping any of those silently breaks the tooltip:
 * with no hover handlers it never opens, and without a ref Radix can't position it. The
 * inner `onClick`/`onKeyDown` compose Radix's injected handler with our drawer-open action.
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

interface JsonPreviewCellProps {
  value: RecordJsonValue | undefined;
  emptyLabel: React.ReactNode;
  /** Open the drawer for this row. */
  onActivate: () => void;
  /** Localized spoken label, e.g. "Open record rec-123 — inputs". */
  accessibleLabel: string;
}

/**
 * Compact one-line JSON preview for inputs/expectations cells. Tooltip shows the
 * structurally-truncated JSON on hover or keyboard focus.
 */
export const JsonPreviewCell = ({ value, emptyLabel, onActivate, accessibleLabel }: JsonPreviewCellProps) => {
  const { theme } = useDesignSystemTheme();
  // `truncateJsonForTooltip` and `renderJsonNode` recursively walk the JSON, and
  // `getJsonColors` returns a new object identity per call. Memoizing all three keeps a
  // 100-row × 2-JSON-column table from rebuilding the tooltip subtree on every parent
  // re-render. Memos run unconditionally — they sit above the empty-value early-return so
  // the hook call order is stable across renders where `value` flips between empty and not.
  const truncated = useMemo(() => (value === undefined ? undefined : truncateJsonForTooltip(value)), [value]);
  const colors = useMemo(
    () => getJsonColors(theme.isDarkMode, theme.colors.textPrimary, theme.colors.textSecondary),
    [theme.isDarkMode, theme.colors.textPrimary, theme.colors.textSecondary],
  );
  const tooltipContent = useMemo(
    () => (truncated === undefined ? null : renderJsonNode(truncated, 0, colors)),
    [truncated, colors],
  );
  if (!value || Object.keys(value).length === 0) {
    return (
      <Typography.Text color="secondary" size="sm">
        {emptyLabel}
      </Typography.Text>
    );
  }
  const compact = JSON.stringify(value);
  return (
    <Tooltip
      componentId="mlflow.eval-datasets-v2.records.cell.json-tooltip"
      content={
        // Override the Du Bois tooltip's auto-inverting surface to `backgroundSecondary`.
        // The default surface is opposite of the body theme (dark in light mode, light in
        // dark mode), which would leave a visibly-mismatched frame around the JSON code
        // block in dark mode and make the dark-mode Prism palette (white / cyan / light
        // blue) unreadable on a light surface. `backgroundSecondary` follows the body
        // theme (light in light mode, dark in dark mode), giving consistent contrast
        // against the syntax-color palette in both themes. The Tooltip component doesn't
        // type-expose `style` as a prop, so we re-skin the tooltip from the inside: a
        // negative margin pulls this div out over the wrapper's own padding
        // (`xs` vertical, `sm` horizontal — see du-bois `getTooltipStyles`), our own
        // padding restores the inset, and a matching border-radius hides the wrapper's
        // original tinted corners. `css` is the only path here — passing it to Tooltip
        // itself would clobber the component's default content styles.
        <div
          css={{
            margin: `-${theme.spacing.xs}px -${theme.spacing.sm}px`,
            padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
            backgroundColor: theme.colors.backgroundSecondary,
            borderRadius: theme.borders.borderRadiusSm,
            fontFamily: 'monospace',
            fontSize: theme.typography.fontSizeSm,
            maxWidth: TOOLTIP_MAX_WIDTH,
            whiteSpace: 'pre-wrap',
            // The Du Bois tooltip renders its arrow as Radix Popper's positioning <span>
            // wrapping the actual <svg><polygon/></svg>, sibling to this content div. The
            // arrow's default `fill: tooltipBackgroundTooltip` lives on that inner SVG, so
            // we target it through the span. Specificity of `.this + span svg` (0,0,1,2)
            // exceeds Du Bois's `.arrow-class` (0,0,1,0), so this overrides without
            // `!important`. Without this, the arrow keeps its inverted color and appears as
            // a mismatched triangle next to the recolored tooltip surface.
            '& + span svg': { fill: theme.colors.backgroundSecondary },
          }}
        >
          {tooltipContent}
        </div>
      }
    >
      {/* Activator fills the cell (display: block, width: 100%) so the inner span's
       * `max-width: 100%` (from truncateCss) resolves against the actual cell width
       * and `text-overflow: ellipsis` fires when the JSON overflows. Pinning the
       * activator to a fixed `maxWidth: 320` instead lets it overflow narrow cells
       * and suppresses ellipsis. */}
      <CellActivator
        componentId="mlflow.eval-datasets-v2.records.cell.json-preview"
        onActivate={onActivate}
        accessibleLabel={accessibleLabel}
        css={{ display: 'block', width: '100%', maxWidth: '100%' }}
      >
        <span css={monospaceCss(theme)}>{compact}</span>
      </CellActivator>
    </Tooltip>
  );
};

interface TagsInlinePreviewBodyProps {
  /** Non-empty list of tag entries. Empty handling lives at the call site so the wrapping
   * table cell decides how to render the "-" placeholder. */
  entries: [string, string][];
  /** Analytics id for the leading `Tag` pill. Differs between the saved-row cell and the
   * phantom-row preview, so it's a prop rather than a constant. */
  componentId: string;
}

/** Pill + "+N more" visual shared by `TagsPreviewCell` (saved rows, wrapped in an
 *  activator + optional tooltip) and the phantom-row tags rendering in
 *  `DatasetRecordsTable` (no activator — phantom has no record yet). */
export const TagsInlinePreviewBody = ({ entries, componentId }: TagsInlinePreviewBodyProps) => {
  const { theme } = useDesignSystemTheme();
  const [firstKey, firstValue] = entries[0];
  const restCount = entries.length - 1;
  return (
    <span css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.xs }}>
      <Tag componentId={componentId} color="default" css={{ maxWidth: 220 }}>
        <Typography.Text ellipsis css={{ maxWidth: 180 }}>
          {firstKey}: {firstValue}
        </Typography.Text>
      </Tag>
      {restCount > 0 && (
        <Typography.Text color="secondary" size="sm">
          <FormattedMessage
            defaultMessage="+{count} more"
            description="Suffix on the dataset record tags column indicating how many additional tags are hidden in the preview"
            values={{ count: restCount }}
          />
        </Typography.Text>
      )}
    </span>
  );
};

interface TagsPreviewCellProps {
  tags: RecordTags | undefined;
  /** Open the drawer for this row. */
  onActivate: () => void;
  /** Localized spoken label, e.g. "Open record rec-123 — tags". */
  accessibleLabel: string;
}

/**
 * Read-only single-line tag preview for the records table. Shows the first tag as a pill
 * plus "+N more" when extra tags are hidden; the full list reveals on hover. Editing has
 * moved to the side panel — clicking anywhere on the cell opens the drawer.
 */
export const TagsPreviewCell = ({ tags, onActivate, accessibleLabel }: TagsPreviewCellProps) => {
  const { theme } = useDesignSystemTheme();
  const entries = tags ? Object.entries(tags) : [];
  if (entries.length === 0) {
    return <Typography.Text color="secondary">-</Typography.Text>;
  }
  const restCount = entries.length - 1;
  const activator = (
    <CellActivator
      componentId="mlflow.eval-datasets-v2.records.cell.tag-preview"
      onActivate={onActivate}
      accessibleLabel={accessibleLabel}
    >
      <TagsInlinePreviewBody entries={entries} componentId="mlflow.eval-datasets-v2.records.cell.tag-preview-pill" />
    </CellActivator>
  );
  // Only wrap in a tooltip when there's something the pill doesn't already surface — i.e.
  // when extra tags are hidden behind "+N more". Single-tag rows skip the tooltip since
  // the pill itself shows the full key/value.
  if (restCount === 0) {
    return activator;
  }
  return (
    <Tooltip
      componentId="mlflow.eval-datasets-v2.records.cell.tag-preview-tooltip"
      content={
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.xs,
            maxWidth: TOOLTIP_MAX_WIDTH,
          }}
        >
          {entries.map(([key, value]) => (
            <span key={key}>
              {key}: {value}
            </span>
          ))}
        </div>
      }
    >
      {activator}
    </Tooltip>
  );
};

interface PlainTextCellProps {
  record: DatasetRecord;
}

interface TimeAgoTextProps {
  iso: string;
}

/**
 * Humanized "X days/hours/minutes ago" cell. The browser-native `title`
 * attribute exposes the full datetime on hover — deliberately *not* the
 * styled Du Bois `Tooltip` used by `<TimeAgo>` itself, to match the spec
 * calling for a native browser tooltip.
 */
const TimeAgoText = ({ iso }: TimeAgoTextProps) => {
  const intl = useIntl();
  const { displayText, tooltipTitle } = getTimeAgoStrings({ date: new Date(iso), intl });
  return (
    <span css={truncateCss} title={tooltipTitle}>
      {displayText}
    </span>
  );
};

export const LastUpdatedCell = ({ record }: PlainTextCellProps) => {
  const value = record.last_update_time ?? record.create_time;
  if (!value) {
    return <Typography.Text color="secondary">-</Typography.Text>;
  }
  return <TimeAgoText iso={value} />;
};

export const CreateTimeCell = ({ record }: PlainTextCellProps) => {
  if (!record.create_time) {
    return <Typography.Text color="secondary">-</Typography.Text>;
  }
  return <TimeAgoText iso={record.create_time} />;
};

export const CreatedByCell = ({ record }: PlainTextCellProps) => {
  if (!record.created_by) {
    return <Typography.Text color="secondary">-</Typography.Text>;
  }
  return <span css={truncateCss}>{record.created_by}</span>;
};

export const LastUpdatedByCell = ({ record }: PlainTextCellProps) => {
  if (!record.last_updated_by) {
    return <Typography.Text color="secondary">-</Typography.Text>;
  }
  return <span css={truncateCss}>{record.last_updated_by}</span>;
};

interface RecordIdCellProps {
  record: DatasetRecord;
  onActivate: () => void;
  accessibleLabel: string;
}

export const RecordIdCell = ({ record, onActivate, accessibleLabel }: RecordIdCellProps) => {
  return (
    <Tooltip
      componentId="mlflow.eval-datasets-v2.records.cell.record-id-tooltip"
      content={<span css={{ maxWidth: TOOLTIP_MAX_WIDTH }}>{record.dataset_record_id}</span>}
    >
      <CellActivator
        componentId="mlflow.eval-datasets-v2.records.cell.record-id"
        onActivate={onActivate}
        accessibleLabel={accessibleLabel}
        css={{ maxWidth: 200 }}
      >
        {/* Typography.Link is used purely for link styling — no href, because the surrounding
         * CellActivator (role="button") is the real interactive element. Nesting an <a> with
         * an href inside a button-role span would create overlapping interactives. */}
        <Typography.Link componentId="mlflow.eval-datasets-v2.records.cell.record-id-link" css={truncateCss}>
          {record.dataset_record_id}
        </Typography.Link>
      </CellActivator>
    </Tooltip>
  );
};

interface SourceCellProps {
  record: DatasetRecord;
}

export const SourceCell = ({ record }: SourceCellProps) => {
  const { source } = record;
  if (source?.trace) {
    return (
      <Tag componentId="mlflow.eval-datasets-v2.records.cell.source.trace" color="indigo">
        <FormattedMessage
          defaultMessage="Trace"
          description="Source-type label for dataset records originating from a trace"
        />
      </Tag>
    );
  }
  if (source?.human) {
    return (
      <Tag componentId="mlflow.eval-datasets-v2.records.cell.source.human" color="teal">
        <FormattedMessage
          defaultMessage="Human"
          description="Source-type label for dataset records authored by a human"
        />
      </Tag>
    );
  }
  if (source?.document) {
    return (
      <Tag componentId="mlflow.eval-datasets-v2.records.cell.source.document" color="lemon">
        <FormattedMessage
          defaultMessage="Document"
          description="Source-type label for dataset records extracted from a document"
        />
      </Tag>
    );
  }
  return <Typography.Text color="secondary">-</Typography.Text>;
};
