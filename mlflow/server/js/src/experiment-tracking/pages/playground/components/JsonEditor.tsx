import { useDesignSystemTheme } from '@databricks/design-system';
import type { ChangeEvent, ReactNode } from 'react';
import { useEffect, useLayoutEffect, useRef } from 'react';

interface Props {
  value: string;
  onChange: (next: string) => void;
  /** Accessible name for the underlying textarea (used by tests and screen readers). */
  ariaLabel: string;
  id?: string;
  placeholder?: string;
  /** Initial visible rows; the editor is vertically resizable from there. */
  minRows?: number;
  /** Tints the border red to flag invalid input. */
  invalid?: boolean;
}

// Editor metrics. Fixed pixel values keep the transparent textarea perfectly
// aligned with the highlighted rows underneath it; both layers must share the
// exact same font, size, line height, and left padding.
const FONT_SIZE = 13;
const LINE_HEIGHT = 20;
const GUTTER_WIDTH = 32;
const GUTTER_GAP = 12;
const PADDING = 12;
const TEXT_PADDING_LEFT = GUTTER_WIDTH + GUTTER_GAP;
const MONO = "'DM Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
// JSON convention (and what our templates / JSON.stringify(…, 2) use).
const INDENT = '  ';

interface TokenColors {
  key: string;
  str: string;
  num: string;
  kw: string;
  punct: string;
  plain: string;
}

// Tuned for legibility on the design-system input background in both themes.
const DARK_COLORS: TokenColors = {
  key: '#83B7E8',
  str: '#6FD08C',
  num: '#E0A85B',
  kw: '#FF7E6B',
  punct: '#8893A0',
  plain: '#C6D0D8',
};
const LIGHT_COLORS: TokenColors = {
  key: '#1F6FEB',
  str: '#117A3D',
  num: '#9A5B00',
  kw: '#C4320A',
  punct: '#57606A',
  plain: '#1F2328',
};

// Matches: (1) a string, (2) an optional trailing colon that marks it as an object key,
// (3) a number, (4) a literal keyword, (5) structural punctuation, (6) any other bare
// token, (7) whitespace. Mirrors the tokenizer in the design spec.
const TOKEN_PATTERN =
  /("(?:\\.|[^"\\])*")(\s*:)?|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)|(true|false|null)\b|([{}[\]:,])|([^\s{}[\]:,"]+)|(\s+)/g;

const highlightLine = (line: string, colors: TokenColors): ReactNode[] => {
  const out: ReactNode[] = [];
  let key = 0;
  // Characters up to here have already been emitted. The tokenizer does not match
  // every possible character (e.g. an in-progress, unbalanced `"`), so we emit any
  // skipped run as plain text — the highlighted layer must reproduce the textarea's
  // text exactly, or characters would appear to vanish (the textarea text is transparent).
  let cursor = 0;
  TOKEN_PATTERN.lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = TOKEN_PATTERN.exec(line)) !== null) {
    // Avoid an infinite loop on a zero-width match (defensive; this pattern always advances).
    if (TOKEN_PATTERN.lastIndex === match.index) {
      TOKEN_PATTERN.lastIndex += 1;
    }
    if (match.index > cursor) {
      out.push(<span key={key++}>{line.slice(cursor, match.index)}</span>);
    }
    if (match[1] !== undefined) {
      const isKey = match[2] !== undefined;
      out.push(
        <span key={key++} css={{ color: isKey ? colors.key : colors.str }}>
          {match[1]}
        </span>,
      );
      if (isKey) {
        out.push(
          <span key={key++} css={{ color: colors.punct }}>
            {match[2]}
          </span>,
        );
      }
    } else if (match[3] !== undefined) {
      out.push(
        <span key={key++} css={{ color: colors.num }}>
          {match[3]}
        </span>,
      );
    } else if (match[4] !== undefined) {
      out.push(
        <span key={key++} css={{ color: colors.kw }}>
          {match[4]}
        </span>,
      );
    } else if (match[5] !== undefined) {
      out.push(
        <span key={key++} css={{ color: colors.punct }}>
          {match[5]}
        </span>,
      );
    } else {
      // Bare tokens and whitespace render in the default text color.
      out.push(match[0]);
    }
    cursor = match.index + match[0].length;
  }
  // Emit any trailing characters the tokenizer left unmatched (e.g. a closing-less quote).
  if (cursor < line.length) {
    out.push(line.slice(cursor));
  }
  return out;
};

/**
 * Lightweight JSON editor with line numbers and syntax highlighting. A
 * transparent `<textarea>` is overlaid on a highlighted rendering of the same
 * text so the caret, selection, and native editing all come from a real
 * textarea while the colored tokens show through underneath. Both layers share
 * identical text metrics so they stay aligned as the content wraps. The editor
 * starts at `minRows` tall and is vertically resizable; content beyond the
 * current height scrolls.
 */
export const JsonEditor = ({ value, onChange, ariaLabel, id, placeholder, minRows = 8, invalid = false }: Props) => {
  const { theme } = useDesignSystemTheme();
  const colors = theme.isDarkMode ? DARK_COLORS : LIGHT_COLORS;
  const lines = value.length > 0 ? value.split('\n') : [''];
  const initialHeight = minRows * LINE_HEIGHT + PADDING * 2;

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  // Selection to restore after a programmatic edit (Tab indent/dedent) once the
  // controlled value has re-rendered.
  const pendingSelection = useRef<[number, number] | null>(null);
  useLayoutEffect(() => {
    if (pendingSelection.current && textareaRef.current) {
      const [selStart, selEnd] = pendingSelection.current;
      textareaRef.current.setSelectionRange(selStart, selEnd);
      pendingSelection.current = null;
    }
  });

  // Keep the latest value/onChange reachable from the document-level listener
  // below without re-subscribing on every keystroke.
  const valueRef = useRef(value);
  valueRef.current = value;
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;

  // Tab indents (Shift+Tab dedents) instead of moving focus; whole lines are
  // (de)indented when a range is selected. This runs as a capture-phase document
  // listener on purpose: the editor lives inside a focus-trapping drawer that
  // intercepts Tab in the capture phase and stops propagation, so a normal
  // (bubble-phase) textarea keydown handler never sees the key.
  useEffect(() => {
    const handleTab = (event: globalThis.KeyboardEvent) => {
      if (event.key !== 'Tab' || event.metaKey || event.ctrlKey || event.altKey) {
        return;
      }
      const el = textareaRef.current;
      if (!el || document.activeElement !== el) {
        return;
      }
      event.preventDefault();
      // Stop the drawer's focus trap from also acting on this Tab.
      event.stopImmediatePropagation();

      const text = valueRef.current;
      const start = el.selectionStart ?? 0;
      const end = el.selectionEnd ?? 0;
      const lineStart = text.lastIndexOf('\n', start - 1) + 1;
      const before = text.slice(0, lineStart);
      const block = text.slice(lineStart, end);
      const after = text.slice(end);

      if (event.shiftKey) {
        let removedFirst = 0;
        let removedTotal = 0;
        const dedented = block
          .split('\n')
          .map((line, index) => {
            const leading = line.match(/^ {1,2}/);
            const removed = leading ? leading[0].length : 0;
            if (index === 0) {
              removedFirst = removed;
            }
            removedTotal += removed;
            return line.slice(removed);
          })
          .join('\n');
        if (removedTotal === 0) {
          return;
        }
        pendingSelection.current = [Math.max(lineStart, start - removedFirst), end - removedTotal];
        onChangeRef.current(before + dedented + after);
        return;
      }

      if (start === end) {
        pendingSelection.current = [start + INDENT.length, start + INDENT.length];
        onChangeRef.current(text.slice(0, start) + INDENT + text.slice(end));
        return;
      }

      const lineCount = block.split('\n').length;
      const indented = block.replace(/^/gm, INDENT);
      pendingSelection.current = [start + INDENT.length, end + INDENT.length * lineCount];
      onChangeRef.current(before + indented + after);
    };
    document.addEventListener('keydown', handleTab, true);
    return () => document.removeEventListener('keydown', handleTab, true);
  }, []);

  const sharedTextCss = {
    fontFamily: MONO,
    fontSize: FONT_SIZE,
    lineHeight: `${LINE_HEIGHT}px`,
    whiteSpace: 'pre-wrap' as const,
    overflowWrap: 'anywhere' as const,
    wordBreak: 'break-word' as const,
  };

  return (
    <div
      css={{
        position: 'relative',
        height: initialHeight,
        minHeight: 3 * LINE_HEIGHT + PADDING * 2,
        resize: 'vertical',
        overflow: 'hidden',
        background: theme.colors.backgroundPrimary,
        border: `1px solid ${invalid ? theme.colors.borderDanger : theme.colors.border}`,
        borderRadius: theme.general.borderRadiusBase,
      }}
    >
      <div css={{ position: 'absolute', inset: 0, overflow: 'auto' }}>
        <div css={{ position: 'relative', minHeight: '100%' }}>
          {/* Full-height gutter background; absolute so it spans the scrolled content. */}
          <div
            aria-hidden
            css={{
              position: 'absolute',
              top: 0,
              bottom: 0,
              left: 0,
              width: GUTTER_WIDTH,
              borderRight: `1px solid ${theme.colors.border}`,
              background: theme.colors.backgroundSecondary,
            }}
          />
          {/* The highlighted rows are in normal flow so they size the content box.
              The textarea below is absolutely positioned to cover that full height,
              which keeps the transparent text, caret, and selection aligned with the
              rows as the editor scrolls. position: relative paints the rows above the
              gutter strip. */}
          <div aria-hidden css={{ position: 'relative', padding: `${PADDING}px ${PADDING}px ${PADDING}px 0` }}>
            {lines.map((line, index) => (
              // eslint-disable-next-line react/no-array-index-key -- lines are positional; no stable id exists
              <div key={index} css={{ display: 'flex', alignItems: 'flex-start' }}>
                <div
                  css={{
                    width: GUTTER_WIDTH,
                    flex: 'none',
                    textAlign: 'right',
                    paddingRight: GUTTER_GAP - 2,
                    color: theme.colors.textSecondary,
                    userSelect: 'none',
                    ...sharedTextCss,
                  }}
                >
                  {index + 1}
                </div>
                <div css={{ flex: 1, minWidth: 0, paddingLeft: GUTTER_GAP, color: colors.plain, ...sharedTextCss }}>
                  {line.length > 0 ? highlightLine(line, colors) : '​'}
                </div>
              </div>
            ))}
          </div>
          <textarea
            ref={textareaRef}
            id={id}
            aria-label={ariaLabel}
            aria-invalid={invalid || undefined}
            value={value}
            spellCheck={false}
            placeholder={placeholder}
            onChange={(event: ChangeEvent<HTMLTextAreaElement>) => onChange(event.target.value)}
            css={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              margin: 0,
              resize: 'none',
              border: 'none',
              outline: 'none',
              background: 'transparent',
              color: 'transparent',
              caretColor: theme.colors.textPrimary,
              padding: `${PADDING}px ${PADDING}px ${PADDING}px ${TEXT_PADDING_LEFT}px`,
              overflow: 'hidden',
              ...sharedTextCss,
              '&::placeholder': { color: theme.colors.textPlaceholder },
            }}
          />
        </div>
      </div>
    </div>
  );
};
