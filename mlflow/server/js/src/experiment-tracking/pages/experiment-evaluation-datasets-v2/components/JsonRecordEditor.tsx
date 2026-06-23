import { useCallback, useEffect, useRef, useState } from 'react';
import Editor, { loader } from '@monaco-editor/react';
import type { Monaco, OnMount } from '@monaco-editor/react';
// The wildcard imports are required: `loader.config({ monaco })` needs the full namespace
// to hand to `@monaco-editor/react`, and `Monaco_.editor.IStandaloneCodeEditor` is only
// reachable via the namespace import. No granular alternative exists.
// eslint-disable-next-line no-restricted-imports
import * as monaco from 'monaco-editor';
// eslint-disable-next-line no-restricted-imports
import type * as Monaco_ from 'monaco-editor';
import { Typography, useDesignSystemTheme } from '@databricks/design-system';

export interface JsonRecordEditorProps {
  /** JSON string. Callers serialize objects via `JSON.stringify(obj, null, 2)`. */
  value: string;
  onChange: (value: string) => void;
  readOnly?: boolean;
  /** CSS length (e.g. "240px") for the minimum editor height. */
  height?: string;
  /**
   * CSS length capping the editor height. The editor grows with its content up to
   * this height, then scrolls internally instead of growing further. Omit for
   * unbounded growth (the default).
   */
  maxHeight?: string;
  /**
   * When true, the editor paints no background of its own so the surrounding
   * surface shows through (useful when embedding inside a card/panel).
   */
  transparentBackground?: boolean;
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

// Hand `@monaco-editor/react` the already-bundled `monaco` module instead of letting it
// fetch the AMD loader from a CDN (or a non-existent local `vs/` path). This is the way
// you use `@monaco-editor/react` together with `MonacoWebpackPlugin` — the plugin handles
// chunking + workers, the loader just consumes the module directly.
loader.config({ monaco });

// Transparent variants of the built-in themes so the editor can blend into a
// surrounding card/panel surface. Registered once at module load (idempotent).
const TRANSPARENT_THEME = { dark: 'mlflow-json-dark-transparent', light: 'mlflow-json-light-transparent' };
const transparentColors = {
  'editor.background': '#00000000',
  'editorGutter.background': '#00000000',
  'minimap.background': '#00000000',
};
monaco.editor.defineTheme(TRANSPARENT_THEME.dark, {
  base: 'vs-dark',
  inherit: true,
  rules: [],
  colors: transparentColors,
});
monaco.editor.defineTheme(TRANSPARENT_THEME.light, { base: 'vs', inherit: true, rules: [], colors: transparentColors });

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
  maxHeight,
  transparentBackground = false,
  ariaLabel,
  errorMessage,
  labelledById,
  describedById,
  onSaveShortcut,
}: JsonRecordEditorProps) => {
  const { theme } = useDesignSystemTheme();
  const hasError = errorMessage !== undefined;
  const maxHeightPx = maxHeight ? heightToPx(maxHeight) : Infinity;
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
      // Grow the wrapper to fit the document, capped at maxHeight (after which the
      // editor scrolls internally). Without maxHeight it grows unbounded.
      const updateHeight = () => {
        const next = Math.min(maxHeightPx, Math.max(heightToPx(height), editor.getContentHeight()));
        setContentHeight(next);
      };
      updateHeight();
      editor.onDidContentSizeChange(updateHeight);
    },
    [height, maxHeightPx],
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
          theme={
            transparentBackground
              ? theme.isDarkMode
                ? TRANSPARENT_THEME.dark
                : TRANSPARENT_THEME.light
              : theme.isDarkMode
                ? 'vs-dark'
                : 'light'
          }
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
            scrollBeyondLastLine: false,
            scrollbar: {
              // Show a vertical scrollbar only when the height is capped; otherwise the
              // wrapper grows to fit and the surrounding container scrolls.
              vertical: maxHeight ? 'auto' : 'hidden',
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
