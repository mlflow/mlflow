import { useEffect, useMemo, useState } from 'react';
import Editor, { loader } from '@monaco-editor/react';
// The loader needs the bundled namespace so it never fetches an editor from a CDN.
// eslint-disable-next-line no-restricted-imports
import * as monaco from 'monaco-editor';
import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { useIntl } from 'react-intl';
import type { FilterFieldDef } from '@databricks/web-shared/traces-table';
import { isSafeFilterKey, metadataFieldRef, tagFieldRef } from '../utils/filterModel';

loader.config({ monaco });

const COLUMNS = [
  'attributes.status',
  'attributes.name',
  'attributes.execution_time_ms',
  'attributes.timestamp_ms',
  'attributes.end_time_ms',
  'attributes.run_id',
  'attributes.request_id',
  'attributes.client_request_id',
  'span.name',
  'span.type',
  'span.status',
  'span.content',
  'request_metadata."mlflow.trace.user"',
  'request_metadata.mlflow.trace.session',
  'request_metadata."mlflow.source.name"',
  'issue.id',
];
const NAMESPACES = new Map(
  Object.entries({
    tag: 'tags',
    metadata: 'request_metadata',
    assessment: 'feedback',
    expectation: 'expectation',
  }),
);

export const SqlFilterEditor = ({
  value,
  onChange,
  fields,
}: {
  value: string;
  onChange: (value: string) => void;
  fields: FilterFieldDef[];
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [editor, setEditor] = useState<monaco.editor.IStandaloneCodeEditor>();
  const columns = useMemo(
    () => [
      ...new Set([
        ...COLUMNS,
        ...fields.flatMap(({ id, keyOptions }) =>
          NAMESPACES.get(id)
            ? (keyOptions ?? [])
                .filter(({ value: key }) => isSafeFilterKey(key))
                .map(({ value: key }) =>
                  id === 'tag'
                    ? tagFieldRef(key)
                    : id === 'metadata'
                      ? metadataFieldRef(key)
                      : `${NAMESPACES.get(id)}.\`${key}\``,
                )
            : [],
        ),
      ]),
    ],
    [fields],
  );
  useEffect(() => {
    if (!editor) return;
    const registration = monaco.languages.registerCompletionItemProvider('sql', {
      triggerCharacters: ['.'],
      provideCompletionItems: (model, position) => {
        if (model !== editor.getModel()) return { suggestions: [] };
        const before = model.getValueInRange({
          startLineNumber: position.lineNumber,
          startColumn: 1,
          endLineNumber: position.lineNumber,
          endColumn: position.column,
        });
        const prefix = before.match(/(?:^|[\s(])((?:[\w.]|`[^`]*`|"[^"]*")*)$/)?.[1] ?? '';
        return {
          suggestions: columns
            .filter((column) => column.toLowerCase().startsWith(prefix.toLowerCase()))
            .map((column) => ({
              label: column,
              insertText: column,
              kind: monaco.languages.CompletionItemKind.Field,
              range: {
                startLineNumber: position.lineNumber,
                endLineNumber: position.lineNumber,
                startColumn: position.column - prefix.length,
                endColumn: position.column,
              },
            })),
        };
      },
    });
    return () => registration.dispose();
  }, [columns, editor]);
  return (
    <>
      <div
        css={{ height: 160, border: `1px solid ${theme.colors.border}`, borderRadius: theme.borders.borderRadiusSm }}
      >
        <Editor
          language="sql"
          value={value}
          onChange={(next) => onChange(next ?? '')}
          theme={theme.isDarkMode ? 'vs-dark' : 'light'}
          onMount={setEditor}
          options={{
            ariaLabel: intl.formatMessage({
              defaultMessage: 'Trace filter predicate',
              description: 'Accessible trace SQL editor label',
            }),
            automaticLayout: true,
            minimap: { enabled: false },
            scrollBeyondLastLine: false,
            wordWrap: 'on',
            fixedOverflowWidgets: false,
            lineNumbersMinChars: 3,
            tabSize: 2,
          }}
        />
      </div>
      <Typography.Text color="secondary">
        {intl.formatMessage({
          defaultMessage:
            'Enter a trace search condition without WHERE, for example: attributes.execution_time_ms > 1000. The server validates syntax, fields, and operators.',
          description: 'Supported OSS trace search predicate syntax',
        })}
      </Typography.Text>
    </>
  );
};
