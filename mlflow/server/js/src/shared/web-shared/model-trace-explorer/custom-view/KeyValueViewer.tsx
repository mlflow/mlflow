import { z } from 'zod';
import { createComponentImplementation } from '@a2ui/react/v0_9';
import { type ComponentApi, DynamicStringSchema } from '@a2ui/web_core/v0_9';

import { CodeSnippetRenderMode } from '../ModelTrace.types';
import { ModelTraceExplorerCodeSnippet } from '../ModelTraceExplorerCodeSnippet';

const FORMATS = ['json', 'text', 'markdown'] as const;
type Format = (typeof FORMATS)[number];

const FORMAT_TO_RENDER_MODE: Record<Format, CodeSnippetRenderMode> = {
  json: CodeSnippetRenderMode.JSON,
  text: CodeSnippetRenderMode.TEXT,
  markdown: CodeSnippetRenderMode.MARKDOWN,
};

/**
 * Schema (API) for the KeyValueViewer component. It renders a SINGLE labeled
 * value using the same snippet renderer as the trace explorer's "Details &
 * Timeline" view: a `label` (key) header plus a value box with the built-in
 * format toggle (Text / JSON / Markdown for string values; an interactive JSON
 * tree for objects/arrays).
 */
export const KeyValueViewerApi = {
  name: 'KeyValueViewer',
  schema: z
    .object({
      label: DynamicStringSchema.describe('The key/label shown above the value, e.g. the attribute name.').optional(),
      value: DynamicStringSchema.describe(
        'JSON-encoded value to display. An object/array renders as a JSON tree; a string can be shown as text/markdown/json.',
      ),
      initialFormat: z
        .enum(FORMATS)
        .describe('Initial display format for string values (json/text/markdown).')
        .optional(),
      hideFormatToggle: z.boolean().describe('When true, hides the per-value format dropdown.').optional(),
    })
    .strict(),
} satisfies ComponentApi;

const asString = (value: unknown): string => (typeof value === 'string' ? value : String(value ?? ''));

// ModelTraceExplorerCodeSnippet does `JSON.parse(data)`, so `data` must be a
// valid JSON string. Values are expected to already be JSON-encoded; if a raw
// (non-JSON) string slips through, encode it so rendering is safe (and so a
// plain string still gets the text/markdown/json toggle).
const toJsonString = (value: unknown): string => {
  const str = asString(value);
  try {
    JSON.parse(str);
    return str;
  } catch {
    return JSON.stringify(str);
  }
};

export const KeyValueViewer = createComponentImplementation(KeyValueViewerApi, ({ props }) => {
  const label = props.label ? asString(props.label) : '';
  const initialRenderMode = props.initialFormat ? FORMAT_TO_RENDER_MODE[props.initialFormat as Format] : undefined;

  // `flex: 1` + `minWidth: 0` make it split evenly (and not overflow) when
  // placed in a Row; harmless when rendered standalone in a block/Column.
  return (
    <div css={{ flex: 1, minWidth: 0 }}>
      <ModelTraceExplorerCodeSnippet
        title={label}
        data={toJsonString(props.value)}
        initialRenderMode={initialRenderMode}
        hideRenderModeDropdown={props.hideFormatToggle === true}
      />
    </div>
  );
});
