import { useCallback, useEffect, useRef } from 'react';
import { Input, Typography, useDesignSystemTheme } from '@databricks/design-system';

export interface JsonRecordEditorProps {
  value: string;
  onChange: (value: string) => void;
  readOnly?: boolean;
  height?: string;
  ariaLabel: string;
  errorMessage?: string;
  labelledById?: string;
  describedById?: string;
  onSaveShortcut?: () => void;
}

/**
 * OSS stub for the dataset record JSON editor.
 *
 * Universe uses a Monaco-backed editor (via `@databricks/editor`). OSS doesn't have that
 * package yet, so we render a plain monospace `<textarea>` with the same prop surface.
 * Users get raw JSON editing — no syntax highlighting, no auto-formatting — but the
 * controlled value/onChange contract is preserved so the side panel's save/discard
 * machinery (and `onSaveShortcut`) all keep working.
 *
 * TODO(oss): wire a real OSS Monaco wrapper (likely reusing the JSON editor under
 * `mlflow/server/js/src/shared/web-shared/model-trace-explorer/`) and remove this stub.
 */
export const JsonRecordEditor = ({
  value,
  onChange,
  readOnly = false,
  height = '240px',
  ariaLabel,
  errorMessage,
  labelledById,
  describedById,
  onSaveShortcut,
}: JsonRecordEditorProps) => {
  const { theme } = useDesignSystemTheme();
  const hasError = errorMessage !== undefined;
  const onSaveShortcutRef = useRef(onSaveShortcut);
  useEffect(() => {
    onSaveShortcutRef.current = onSaveShortcut;
  }, [onSaveShortcut]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 's') {
      e.preventDefault();
      onSaveShortcutRef.current?.();
    }
  }, []);

  return (
    <div
      role={labelledById || describedById ? 'group' : undefined}
      aria-labelledby={labelledById}
      aria-describedby={describedById}
    >
      <Input.TextArea
        componentId="mlflow.eval-datasets-v2.json-editor.textarea"
        aria-label={ariaLabel}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={readOnly}
        css={{
          fontFamily: '"SF Mono", Menlo, Consolas, monospace',
          fontSize: theme.typography.fontSizeSm,
          minHeight: height,
          border: `1px solid ${hasError ? theme.colors.borderDanger : theme.colors.border}`,
          borderRadius: theme.borders.borderRadiusSm,
        }}
        autoSize={{ minRows: 10 }}
      />
      {hasError && (
        <div role="alert" css={{ marginTop: theme.spacing.xs }}>
          <Typography.Text size="sm" color="error">
            {errorMessage}
          </Typography.Text>
        </div>
      )}
    </div>
  );
};
