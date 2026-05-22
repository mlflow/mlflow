import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  useFixDefaultKeybindingsWithMultipleEditorsPlugin,
  useReadOnlyPlugin,
  useRemoveCommandPalettePlugin,
  useTrackChangesPlugin,
} from '@databricks/editor/commonPlugins';
import type {
  EditorModule,
  ICodeEditorPlugin,
  IStandaloneCodeEditor,
  IStandaloneEditorConstructionOptions,
} from '@databricks/editor/unifiedEditor';
import { getEditor, getMonacoApi } from '@databricks/editor/unifiedEditor';
import { Typography, useDesignSystemTheme } from '@databricks/design-system';

export interface JsonRecordEditorProps {
  /** JSON string. Caller is responsible for serializing objects via JSON.stringify(..., null, 2). */
  value: string;
  onChange: (value: string) => void;
  readOnly?: boolean;
  /** Editor height. Strings (e.g. "240px") or CSS lengths. */
  height?: string;
  ariaLabel: string;
  /**
   * Localized error message to render below the editor. Also tints the editor border red so
   * the user can spot the offending field at a glance. Pass `undefined` for the no-error state.
   */
  errorMessage?: string;
  /**
   * Id of a visible element that labels this editor. When supplied, the outer wrapper becomes
   * a `role="group"` whose accessible name is the labelled element's text — the cleanest way
   * to bind a bare-text label (e.g. a section's "Inputs"/"Expectations" title) to a Monaco
   * editor whose inner textarea id is not exposed.
   */
  labelledById?: string;
  /** Id of a visible element that describes this editor (e.g. a `FormUI.Hint`). */
  describedById?: string;
  /**
   * Called when Cmd/Ctrl-S is pressed *while the editor has focus*. Monaco's contenteditable
   * swallows keydown events before they bubble to ancestor handlers, so the surrounding
   * drawer's `onKeyDown` listener never fires for editor-focused saves — registering an
   * editor action is the only way to intercept the shortcut.
   */
  onSaveShortcut?: () => void;
}

const MODULE_NAME: EditorModule = 'mlflowEvalDatasetsJsonEditor';
const noPlugins: ICodeEditorPlugin[] = [];
const noop = () => {};

/**
 * Monaco-backed JSON editor for dataset record inputs/expectations. Mirrors the
 * LakewatchJsonEditor pattern: get editor + apply track-changes plugin for controlled
 * value/onChange. `detectIndentation` + JSON language give the IDE feel.
 *
 * Note: this component itself imports `@databricks/editor/*` directly. Callers should
 * wrap it in React.lazy via `LazyJsonRecordEditor` so Monaco stays out of the main bundle.
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
  const { UnifiedEditor } = getEditor();
  const [editorInstance, setEditorInstance] = useState<IStandaloneCodeEditor>();
  // Mirror Monaco's content height so the wrapper grows with the JSON. The side panel itself
  // handles scrolling — the editor must not capture wheel events, hence the scrollbar config
  // in defaultOptions below.
  const [contentHeight, setContentHeight] = useState<number | null>(null);

  const { defaultOptions: readOnlyOptions } = useReadOnlyPlugin({ editorInstance, readOnly });

  useTrackChangesPlugin({
    editorInstance,
    value,
    onValueChange: onChange ?? noop,
    enableFastTypingFix: true,
  });
  useRemoveCommandPalettePlugin({ editorInstance });
  useFixDefaultKeybindingsWithMultipleEditorsPlugin(MODULE_NAME, editorInstance);

  // Bind Cmd/Ctrl-S inside Monaco so the user can save without first tabbing out of the
  // editor. The ref dance lets us register the action once per editor instance while
  // always invoking the freshest callback (re-renders update the ref synchronously).
  const onSaveShortcutRef = useRef(onSaveShortcut);
  useEffect(() => {
    onSaveShortcutRef.current = onSaveShortcut;
  }, [onSaveShortcut]);
  useEffect(() => {
    if (!editorInstance) return undefined;
    const monaco = getMonacoApi();
    const action = editorInstance.addAction({
      id: 'mlflow.eval-datasets-v2.json-editor.save',
      label: 'Save',
      keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS],
      run: () => {
        onSaveShortcutRef.current?.();
      },
    });
    return () => action.dispose();
  }, [editorInstance]);

  useEffect(() => {
    if (!editorInstance) return undefined;
    const update = () => setContentHeight(editorInstance.getContentHeight());
    update();
    const disposable = editorInstance.onDidContentSizeChange(update);
    return () => disposable.dispose();
  }, [editorInstance]);

  // Initial value must be captured at mount time — Monaco read-only editors don't show
  // initial value otherwise. Subsequent value changes flow through useTrackChangesPlugin.
  const initialValueRef = useRef(value ?? '');

  const handleEditorInstanceChange = useCallback((editor: IStandaloneCodeEditor | undefined) => {
    setEditorInstance(editor);
  }, []);

  const defaultOptions: IStandaloneEditorConstructionOptions = useMemo(
    () => ({
      value: initialValueRef.current,
      language: 'json',
      ariaLabel,
      automaticLayout: true,
      contextmenu: false,
      folding: true,
      glyphMargin: false,
      lineDecorationsWidth: 0,
      minimap: { enabled: false },
      occurrencesHighlight: 'off',
      padding: { top: 8, bottom: 8 },
      tabSize: 2,
      wordWrap: 'on',
      detectIndentation: true,
      // Pretty-print pasted minified JSON and auto-indent as the user types.
      // `autoIndent: 'full'` is inherited from UNIFIED_CODE_EDITOR_DEFAULT_OPTIONS.
      formatOnPaste: true,
      formatOnType: true,
      lineNumbersMinChars: 3,
      // The wrapper grows to fit content (see contentHeight effect), so disable internal
      // scrollbars and let wheel events bubble to the side panel instead of being swallowed.
      scrollBeyondLastLine: false,
      scrollbar: {
        vertical: 'hidden',
        horizontal: 'hidden',
        alwaysConsumeMouseWheel: false,
      },
      // Both themes are eagerly registered by UnifiedThemePlugin, so we can switch by
      // string name without registering anything ourselves. Matches the pattern in
      // js/packages/editor/src/codeBlock/ReadOnlyCodeBlock.tsx.
      theme: theme.isDarkMode ? 'databricks-unified-dark' : 'databricks-unified-light',
      ...readOnlyOptions,
    }),
    [ariaLabel, readOnlyOptions, theme.isDarkMode],
  );

  return (
    <div
      role={labelledById || describedById ? 'group' : undefined}
      aria-labelledby={labelledById}
      aria-describedby={describedById}
    >
      <div
        css={{
          height: contentHeight != null ? contentHeight : height,
          border: `1px solid ${hasError ? theme.colors.borderDanger : theme.colors.border}`,
          borderRadius: theme.borders.borderRadiusSm,
          overflow: 'hidden',
        }}
      >
        <UnifiedEditor
          type="codeEditor"
          plugins={noPlugins}
          onEditorInstanceChange={handleEditorInstanceChange}
          defaultOptions={defaultOptions}
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
