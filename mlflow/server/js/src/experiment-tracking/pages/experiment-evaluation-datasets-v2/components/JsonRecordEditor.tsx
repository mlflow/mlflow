import { useCallback, useEffect, useRef, useState } from 'react';
import Editor, { loader } from '@monaco-editor/react';
import type { Monaco, OnMount } from '@monaco-editor/react';
import type * as Monaco_ from 'monaco-editor';
import { Typography, useDesignSystemTheme } from '@databricks/design-system';

export interface JsonRecordEditorProps {
  /** JSON string. Callers serialize objects via `JSON.stringify(obj, null, 2)`. */
  value: string;
  onChange: (value: string) => void;
  readOnly?: boolean;
  /** CSS length (e.g. "240px") for the minimum editor height. */
  height?: string;
  ariaLabel: string;
  /** Localized error string; renders below the editor and tints the border red. */
  errorMessage?: string;
  labelledById?: string;
  describedById?: string;
  /**
   * Called when Cmd/Ctrl-S is pressed while the editor has focus. Monaco swallows
   * keydown events before they bubble, so we register an editor action instead.
   */
  onSaveShortcut?: () => void;
}

// Configure the loader once at module-load to use the locally-bundled monaco-editor copy
// (shipped by `monaco-editor-webpack-plugin`) instead of fetching from the CDN. Without this
// the editor 404s in air-gapped environments and slows the initial open by ~500ms otherwise.
loader.config({ paths: { vs: '/static-files/static/js/vs' } });

const heightToPx = (h: string): number => {
  const n = parseInt(h, 10);
  return Number.isNaN(n) ? 240 : n;
};

/**
 * Monaco-backed JSON editor for dataset record `inputs` / `expectations`. Self-contained:
 * loads its own Monaco bundle (lazy-split by webpack via MonacoWebpackPlugin) so callers
 * don't need a separate lazy wrapper aside from `LazyJsonRecordEditor` for the React.lazy
 * code-splitting boundary.
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
  const editorRef = useRef<Monaco_.editor.IStandaloneCodeEditor | null>(null);
  const [contentHeight, setContentHeight] = useState<number>(heightToPx(height));

  // Stash the latest shortcut handler so the editor action always invokes the freshest
  // closure without forcing re-registration on every parent render.
  const onSaveShortcutRef = useRef(onSaveShortcut);
  useEffect(() => {
    onSaveShortcutRef.current = onSaveShortcut;
  }, [onSaveShortcut]);

  const handleMount: OnMount = useCallback(
    (editor: Monaco_.editor.IStandaloneCodeEditor, monaco: Monaco) => {
      editorRef.current = editor;
      editor.addAction({
        id: 'mlflow.eval-datasets-v2.json-editor.save',
        label: 'Save',
        keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS],
        run: () => onSaveShortcutRef.current?.(),
      });
      // Grow the wrapper to fit the document; the side panel itself handles scrolling.
      const updateHeight = () => {
        const next = Math.max(heightToPx(height), editor.getContentHeight());
        setContentHeight(next);
      };
      updateHeight();
      editor.onDidContentSizeChange(updateHeight);
    },
    [height],
  );

  return (
    <div
      role={labelledById || describedById ? 'group' : undefined}
      aria-labelledby={labelledById}
      aria-describedby={describedById}
    >
      <div
        css={{
          height: contentHeight,
          border: `1px solid ${hasError ? theme.colors.borderDanger : theme.colors.border}`,
          borderRadius: theme.borders.borderRadiusSm,
          overflow: 'hidden',
        }}
      >
        <Editor
          language="json"
          value={value}
          onChange={(next) => onChange(next ?? '')}
          theme={theme.isDarkMode ? 'vs-dark' : 'light'}
          onMount={handleMount}
          options={{
            readOnly,
            ariaLabel,
            automaticLayout: true,
            contextmenu: false,
            folding: true,
            glyphMargin: false,
            lineDecorationsWidth: 0,
            minimap: { enabled: false },
            padding: { top: 8, bottom: 8 },
            tabSize: 2,
            wordWrap: 'on',
            detectIndentation: true,
            formatOnPaste: true,
            formatOnType: true,
            lineNumbersMinChars: 3,
            // Wrapper grows to fit content via the onDidContentSizeChange listener above;
            // keep Monaco's own scrollbars hidden and let mousewheel bubble to the side panel.
            scrollBeyondLastLine: false,
            scrollbar: {
              vertical: 'hidden',
              horizontal: 'hidden',
              alwaysConsumeMouseWheel: false,
            },
          }}
        />
      </div>
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
