import type { ZodTypeAny } from 'zod';
import {
  type A2uiMessage,
  CreateSurfaceMessageSchema,
  UpdateComponentsMessageSchema,
  UpdateDataModelMessageSchema,
} from '@a2ui/web_core/v0_9';

import { AssessmentBoardApi } from '../AssessmentBoard';
import { AssessmentCardApi } from '../AssessmentCard';
import { CarouselApi } from '../Carousel';
import { ContentViewerApi } from '../ContentViewer';
import { DataTableApi } from '../DataTable';
import { FeedbackFormApi } from '../FeedbackForm';
import { StatCardApi } from '../StatCard';
import { TimelineChartApi } from '../TimelineChart';
import { TreeViewApi } from '../TreeView';

// Per-component prop schemas for the custom catalog components. The basic
// catalog components (Text/Row/Column) are intentionally absent: they're
// validated by the renderer at bind time, and we let them pass through here so
// the LLM can still lay out content with rows/columns.
const COMPONENT_SCHEMAS: Record<string, ZodTypeAny> = {
  [StatCardApi.name]: StatCardApi.schema,
  [DataTableApi.name]: DataTableApi.schema,
  [TimelineChartApi.name]: TimelineChartApi.schema,
  [TreeViewApi.name]: TreeViewApi.schema,
  [CarouselApi.name]: CarouselApi.schema,
  [FeedbackFormApi.name]: FeedbackFormApi.schema,
  [ContentViewerApi.name]: ContentViewerApi.schema,
  [AssessmentBoardApi.name]: AssessmentBoardApi.schema,
  [AssessmentCardApi.name]: AssessmentCardApi.schema,
};

export type ValidateResult =
  | { ok: true; messages: A2uiMessage[] }
  | { ok: false; error: string };

type RawMessage = Record<string, unknown>;

// Pulls a message array out of whatever JSON the model returned. We accept the
// canonical encodings: a bare array of messages, a `{ messages: [...] }`
// wrapper, or a single message object.
const toMessageArray = (raw: unknown): RawMessage[] | undefined => {
  if (Array.isArray(raw)) {
    return raw as RawMessage[];
  }
  if (raw && typeof raw === 'object') {
    const obj = raw as Record<string, unknown>;
    if (Array.isArray(obj.messages)) {
      return obj.messages as RawMessage[];
    }
    // A single message object (has one of the known top-level keys).
    if ('createSurface' in obj || 'updateComponents' in obj || 'updateDataModel' in obj) {
      return [obj as RawMessage];
    }
  }
  return undefined;
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === 'object' && !Array.isArray(value);

// Validates the props of a single component against the custom catalog schema.
// `id` and `component` are stripped first since the per-component schemas (like
// the renderer) only describe the component's own props.
const validateComponentProps = (component: Record<string, unknown>): string | undefined => {
  const componentName = component.component;
  if (typeof componentName !== 'string') {
    return 'A component is missing its string "component" type.';
  }
  if (component.id === undefined || component.id === null || component.id === '') {
    return `Component "${componentName}" is missing a non-empty "id".`;
  }
  const schema = COMPONENT_SCHEMAS[componentName];
  if (!schema) {
    // Basic catalog component (Text/Row/Column) or unknown — let it pass; the
    // renderer/binder is the source of truth for these.
    return undefined;
  }
  const { id: _id, component: _component, ...props } = component;
  const result = schema.safeParse(props);
  if (!result.success) {
    const detail = result.error.issues.map((issue) => `${issue.path.join('.') || '(root)'}: ${issue.message}`).join('; ');
    return `Component "${String(component.id)}" (${componentName}) has invalid props: ${detail}`;
  }
  return undefined;
};

/**
 * Validates and normalizes an LLM-generated A2UI message stream so it can be
 * safely handed to `MessageProcessor.processMessages` (which does NOT validate
 * against the catalog). We:
 *
 *  - extract the message array from the model's JSON (array / wrapper / single),
 *  - drop any `createSurface` / `deleteSurface` the model emitted and inject our
 *    own `createSurface` so the surface id + catalog id are host-controlled
 *    (the model shouldn't pick surface ids or delete surfaces),
 *  - rewrite the `surfaceId` on every kept message to the target surface,
 *  - validate each message envelope (Zod) and each custom component's props,
 *  - require at least one `updateComponents` containing a `root` component.
 */
export const validateAndPrepareMessages = (
  raw: unknown,
  { surfaceId, catalogId }: { surfaceId: string; catalogId: string },
): ValidateResult => {
  const rawMessages = toMessageArray(raw);
  if (!rawMessages || rawMessages.length === 0) {
    return { ok: false, error: 'The model did not return any A2UI messages.' };
  }

  const kept: A2uiMessage[] = [
    {
      version: 'v0.9',
      createSurface: { surfaceId, catalogId, sendDataModel: true },
    },
  ];

  let sawRoot = false;

  for (const message of rawMessages) {
    if (!isRecord(message)) {
      return { ok: false, error: 'Encountered a message that is not a JSON object.' };
    }
    // The model controls surface lifecycle for itself; we own it. Skip its
    // createSurface/deleteSurface entirely.
    if ('createSurface' in message || 'deleteSurface' in message) {
      continue;
    }

    if ('updateComponents' in message) {
      const payload = isRecord(message.updateComponents) ? { ...message.updateComponents, surfaceId } : undefined;
      const normalized = { version: 'v0.9', updateComponents: payload };
      const parsed = UpdateComponentsMessageSchema.safeParse(normalized);
      if (!parsed.success) {
        return {
          ok: false,
          error: `Invalid updateComponents message: ${parsed.error.issues.map((i) => i.message).join('; ')}`,
        };
      }
      const components = (payload?.components ?? []) as Record<string, unknown>[];
      for (const component of components) {
        if (!isRecord(component)) {
          return { ok: false, error: 'A component entry is not a JSON object.' };
        }
        if (component.id === 'root') {
          sawRoot = true;
        }
        const componentError = validateComponentProps(component);
        if (componentError) {
          return { ok: false, error: componentError };
        }
      }
      kept.push(parsed.data);
      continue;
    }

    if ('updateDataModel' in message) {
      const payload = isRecord(message.updateDataModel) ? { ...message.updateDataModel, surfaceId } : undefined;
      const normalized = { version: 'v0.9', updateDataModel: payload };
      const parsed = UpdateDataModelMessageSchema.safeParse(normalized);
      if (!parsed.success) {
        return {
          ok: false,
          error: `Invalid updateDataModel message: ${parsed.error.issues.map((i) => i.message).join('; ')}`,
        };
      }
      kept.push(parsed.data);
      continue;
    }

    // Unknown / unsupported message shape — ignore it rather than fail the whole
    // generation, since the processor would reject it anyway.
  }

  // Sanity-check our own injected createSurface against the schema too.
  const surfaceCheck = CreateSurfaceMessageSchema.safeParse(kept[0]);
  if (!surfaceCheck.success) {
    return { ok: false, error: 'Failed to construct a valid createSurface message.' };
  }

  if (!sawRoot) {
    return { ok: false, error: 'The generated UI has no "root" component to render.' };
  }

  return { ok: true, messages: kept };
};
