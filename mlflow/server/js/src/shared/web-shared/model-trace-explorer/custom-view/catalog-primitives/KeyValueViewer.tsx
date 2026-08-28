import { z } from 'zod';
import { createComponentImplementation, type ReactComponentImplementation } from '@a2ui/react/v0_9';
import { type ComponentApi, DynamicStringSchema } from '@a2ui/web_core/v0_9';

import { CodeSnippetRenderMode } from '../../ModelTrace.types';
import { ModelTraceExplorerCodeSnippet } from '../../ModelTraceExplorerCodeSnippet';
import { asString } from '../catalogPrimitiveUtils';

const FORMATS = ['json', 'text', 'markdown'] as const;
type Format = (typeof FORMATS)[number];

const FORMAT_TO_RENDER_MODE: Record<Format, CodeSnippetRenderMode> = {
  json: CodeSnippetRenderMode.JSON,
  text: CodeSnippetRenderMode.TEXT,
  markdown: CodeSnippetRenderMode.MARKDOWN,
};

const isFormat = (value: unknown): value is Format =>
  typeof value === 'string' && (FORMATS as readonly string[]).includes(value);

/**
 * Schema (API) for the KeyValueViewer component.
 */
const KeyValueViewerApi = {
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
      weight: z.number().describe('Relative flex weight when placed directly inside a Row/Column.').optional(),
    })
    .strict(),
} satisfies ComponentApi;

// ModelTraceExplorerCodeSnippet expects `data` to be a valid JSON string (it
// runs JSON.parse on it) and renders a JSON tree when it parses to a non-string.
// `value` is a DynamicString, so at runtime it can resolve to a real object/array
// (e.g. bound to a data-model path), not just a JSON-encoded string. Serialize
// objects/arrays directly so they render as a tree instead of "[object Object]".
const toJsonString = (value: unknown): string => {
  if (typeof value === 'object' && value !== null) {
    try {
      return JSON.stringify(value);
    } catch {
      // Non-serializable (e.g. circular): keep the output valid JSON so the
      // downstream JSON.parse doesn't throw.
      return JSON.stringify(String(value));
    }
  }
  const str = asString(value);
  try {
    JSON.parse(str);
    return str;
  } catch {
    return JSON.stringify(str);
  }
};

export const KeyValueViewer: ReactComponentImplementation = createComponentImplementation(
  KeyValueViewerApi,
  ({ props }) => {
    const label = props.label ? asString(props.label) : '';
    const initialRenderMode = isFormat(props.initialFormat) ? FORMAT_TO_RENDER_MODE[props.initialFormat] : undefined;
    const weight = typeof props.weight === 'number' ? props.weight : 1;

    return (
      <div css={{ flex: `${weight}`, minWidth: 0 }}>
        <ModelTraceExplorerCodeSnippet
          title={label}
          data={toJsonString(props.value)}
          initialRenderMode={initialRenderMode}
          hideRenderModeDropdown={props.hideFormatToggle === true}
        />
      </div>
    );
  },
);
