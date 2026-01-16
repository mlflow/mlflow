import React, { useRef, useCallback, useEffect, useState, useMemo } from 'react';
import { useDesignSystemTheme } from '@databricks/design-system';

// Pattern for template variables like {{ inputs }}, {{ outputs }}, etc.
const TEMPLATE_VARIABLE_PATTERN = /(\{\{\s*[a-zA-Z_][a-zA-Z0-9_]*\s*\}\})/g;

interface HighlightedTextAreaProps {
  value: string;
  onChange: (value: string) => void;
  onBlur?: () => void;
  placeholder?: string;
  readOnly?: boolean;
  rows?: number;
  id?: string;
  name?: string;
}

/**
 * A textarea with synchronized backdrop highlighting for template variables.
 *
 * Uses the overlay pattern:
 * - Transparent textarea on top handles all input
 * - Backdrop div behind renders highlighted content
 * - Both must have identical styling for alignment
 */
export const HighlightedTextArea: React.FC<HighlightedTextAreaProps> = ({
  value,
  onChange,
  onBlur,
  placeholder,
  readOnly = false,
  rows = 7,
  id,
  name,
}) => {
  const { theme } = useDesignSystemTheme();
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const backdropRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isFocused, setIsFocused] = useState(false);

  // Sync scroll position between textarea and backdrop
  const handleScroll = useCallback(() => {
    if (textareaRef.current && backdropRef.current) {
      backdropRef.current.scrollTop = textareaRef.current.scrollTop;
      backdropRef.current.scrollLeft = textareaRef.current.scrollLeft;
    }
  }, []);

  // Handle resize observer to keep backdrop in sync
  useEffect(() => {
    const textarea = textareaRef.current;
    const backdrop = backdropRef.current;

    if (!textarea || !backdrop) return;

    const resizeObserver = new ResizeObserver(() => {
      // Sync dimensions when textarea is resized
      backdrop.style.width = `${textarea.offsetWidth}px`;
      backdrop.style.height = `${textarea.offsetHeight}px`;
    });

    resizeObserver.observe(textarea);

    return () => resizeObserver.disconnect();
  }, []);

  const highlightedContent = useMemo(() => {
    if (!value) {
      return (
        <span
          css={{
            color: theme.colors.textPlaceholder,
          }}
        >
          {placeholder}
        </span>
      );
    }

    const parts = value.split(TEMPLATE_VARIABLE_PATTERN);
    return parts.map((part, index) => {
      if (TEMPLATE_VARIABLE_PATTERN.test(part)) {
        // Reset regex lastIndex since we're using it multiple times
        TEMPLATE_VARIABLE_PATTERN.lastIndex = 0;
        return (
          <mark
            key={index}
            css={{
              backgroundColor: theme.colors.tagDefault,
              color: theme.colors.textPrimary,
              borderRadius: theme.borders.borderRadiusSm,
              padding: '1px 2px',
              margin: '0 -2px', // Compensate for padding to maintain text flow
            }}
          >
            {part}
          </mark>
        );
      }
      // Preserve whitespace and newlines
      return <span key={index}>{part}</span>;
    });
  }, [value, placeholder, theme]);

  // Shared styles for both textarea and backdrop to ensure perfect alignment
  const sharedStyles = {
    fontFamily: 'inherit',
    fontSize: theme.typography.fontSizeBase,
    lineHeight: theme.typography.lineHeightBase,
    padding: '8px 12px',
    border: `1px solid transparent`,
    borderRadius: theme.borders.borderRadiusMd,
    width: '100%',
    boxSizing: 'border-box' as const,
    wordWrap: 'break-word' as const,
    whiteSpace: 'pre-wrap' as const,
    overflowWrap: 'break-word' as const,
  };

  return (
    <div
      ref={containerRef}
      css={{
        position: 'relative',
        width: '100%',
      }}
    >
      {/* Backdrop with highlighted content */}
      <div
        ref={backdropRef}
        aria-hidden="true"
        css={{
          ...sharedStyles,
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          overflow: 'hidden',
          pointerEvents: 'none',
          backgroundColor: readOnly ? theme.colors.actionDisabledBackground : theme.colors.backgroundPrimary,
          color: theme.colors.textPrimary,
          border: `1px solid ${isFocused ? theme.colors.actionPrimaryBackgroundDefault : theme.colors.borderDecorative}`,
          // Match textarea's default height
          minHeight: `${rows * 1.5}em`,
        }}
      >
        {highlightedContent}
      </div>

      {/* Transparent textarea on top for input */}
      <textarea
        ref={textareaRef}
        id={id}
        name={name}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onBlur={() => {
          setIsFocused(false);
          onBlur?.();
        }}
        onFocus={() => setIsFocused(true)}
        onScroll={handleScroll}
        placeholder={placeholder}
        readOnly={readOnly}
        rows={rows}
        css={{
          ...sharedStyles,
          position: 'relative',
          backgroundColor: 'transparent',
          color: 'transparent',
          caretColor: theme.colors.textPrimary,
          resize: 'vertical',
          minHeight: `${rows * 1.5}em`,
          // Ensure textarea is on top
          zIndex: 1,
          // Hide placeholder since backdrop shows it
          '&::placeholder': {
            color: 'transparent',
          },
          '&:focus': {
            outline: 'none',
          },
          cursor: readOnly ? 'default' : 'text',
        }}
      />
    </div>
  );
};
