import { z, type ZodTypeAny } from 'zod';
import {
  type A2uiMessage,
  ChildListSchema,
  CreateSurfaceMessageSchema,
  DynamicStringSchema,
  UpdateComponentsMessageSchema,
  UpdateDataModelMessageSchema,
} from '@a2ui/web_core/v0_9';

import { AssessmentBoard } from '../catalog-primitives/AssessmentBoard';
import { AssessmentCard } from '../catalog-primitives/AssessmentCard';
import { Card } from '../catalog-primitives/Card';
import { Icon } from '../catalog-primitives/Icon';
import { KeyValueViewer } from '../catalog-primitives/KeyValueViewer';
import { Markdown } from '../catalog-primitives/Markdown';
import { StatCard } from '../catalog-primitives/StatCard';
import {
  SPAN_FIELD_SOURCE_NAME,
  isKnownSource,
  isRecord,
  isSourceMarker,
  isSpanRefMarker,
  isStructuralSource,
  isValidSpanFieldMarker,
  isValidSpanRefSelector,
  unwrapSpanRefSelector,
} from '../customViewSources';

// Each catalog primitive's component implementation carries its A2UI `name` and
// Zod `schema` (via `createComponentImplementation`), so we derive the validation
// map straight from them — the component impls are the single source of truth, so
// the primitives don't need to export their schema definitions separately.
const COMPONENT_SCHEMAS: Record<string, ZodTypeAny> = Object.fromEntries(
  [StatCard, Icon, Card, Markdown, AssessmentBoard, AssessmentCard, KeyValueViewer].map(
    (component): [string, ZodTypeAny] => [component.name, component.schema],
  ),
);

// Prop shapes for the A2UI "basic catalog" layout primitives (Text / Row /
// Column). Unlike the custom primitives above, these have no local
// `ReactComponentImplementation` whose Zod schema we can reuse, so we declare
// their allowlisted prop shapes here.
//
// These MUST mirror the props the A2UI basic-catalog schema (and its React
// renderer) actually accept — NOT merely the subset documented to the model in
// `CATALOG_REFERENCE` (buildAgentPrompt.ts). This runs `.strict()` at per-trace
// render time, so anything the renderer legitimately supports but we omit here
// would make a valid saved view fail to render (error placeholder). `justify`
// on Row/Column and `weight` (the shared `CatalogComponentCommon` flex weight)
// are renderer props even though the prompt doesn't teach all of them; the
// custom `Text` renderer also reads `variant` / `weight`. `.strict()` still
// blocks any prop outside this set — no unknown-field passthrough.
const TEXT_VARIANTS = ['h1', 'h2', 'h3', 'h4', 'h5', 'caption', 'body'] as const;
const LAYOUT_ALIGNMENTS = ['start', 'center', 'end', 'stretch'] as const;
// Main-axis distribution enum shared by Row and Column in the basic catalog.
const LAYOUT_JUSTIFY = ['start', 'center', 'end', 'spaceBetween', 'spaceAround', 'spaceEvenly', 'stretch'] as const;

const BASIC_COMPONENT_SCHEMAS = {
  Text: z
    .object({
      text: DynamicStringSchema,
      variant: z.enum(TEXT_VARIANTS).optional(),
      weight: z.number().optional(),
    })
    .strict(),
  Row: z
    .object({
      children: ChildListSchema,
      align: z.enum(LAYOUT_ALIGNMENTS).optional(),
      justify: z.enum(LAYOUT_JUSTIFY).optional(),
      weight: z.number().optional(),
    })
    .strict(),
  Column: z
    .object({
      children: ChildListSchema,
      align: z.enum(LAYOUT_ALIGNMENTS).optional(),
      justify: z.enum(LAYOUT_JUSTIFY).optional(),
      weight: z.number().optional(),
    })
    .strict(),
} satisfies Record<string, ZodTypeAny>;

// The complete, CLOSED allowlist of catalog component types (custom + basic).
// The validator is the only gate before the A2UI renderer, which does NOT
// re-check components against the catalog — so any component type absent from
// this map (an injected `iframe`/`script`, a typo, an unlisted primitive) must
// be rejected rather than forwarded to the DOM.
//
// Built on a NULL prototype so a component named after an `Object.prototype`
// member (`constructor`, `toString`, `__proto__`, `hasOwnProperty`, …) can't
// slip through: with a normal prototype, `'constructor' in schemas` is `true`
// (bypassing the template-time allowlist) and `schemas['constructor']` resolves
// to `Object`'s constructor — truthy but with no `.safeParse`, which would throw
// at per-trace validation instead of cleanly rejecting.
const CATALOG_COMPONENT_SCHEMAS: Record<string, ZodTypeAny> = Object.assign(
  Object.create(null),
  COMPONENT_SCHEMAS,
  BASIC_COMPONENT_SCHEMAS,
);

// The A2UI protocol version this custom-view integration targets. Every emitted
// message carries it, and the authoring prompt references it, so keep it single-sourced.
export const A2UI_VERSION = 'v0.9';

export type ValidateResult = { ok: true; messages: A2uiMessage[] } | { ok: false; error: string };

// Pulls a message array out of whatever JSON the model returned. We accept the
// canonical encodings: a bare array, a `{ messages: [...] }` wrapper, or a
// single message object. Entries stay `unknown` — each caller guards them with
// `isRecord` before use, so we never assert a shape the model hasn't proven.
const toMessageArray = (raw: unknown): unknown[] | undefined => {
  if (Array.isArray(raw)) {
    return raw;
  }
  if (isRecord(raw)) {
    if (Array.isArray(raw['messages'])) {
      return raw['messages'];
    }
    // A single message object (has one of the known top-level keys).
    if ('createSurface' in raw || 'updateComponents' in raw || 'updateDataModel' in raw) {
      return [raw];
    }
  }
  return undefined;
};

// Models occasionally emit a component as `{ id, component, props: { ... } }`
// (React-style) instead of our flat shape where every prop sits directly on the
// object alongside `id`/`component`. Hoist a nested `props` object up so both the
// strict per-component schema and the renderer (which consume the flat shape)
// see the real props. Existing top-level keys win over the nested ones.
const flattenComponentProps = (component: Record<string, unknown>): Record<string, unknown> => {
  if (!isRecord(component['props'])) {
    return component;
  }
  const { props, ...rest } = component;
  return { ...(props as Record<string, unknown>), ...rest };
};

// Validates the props of a single component against the catalog schema.
// `id` and `component` are stripped first since the per-component schemas (like
// the renderer) only describe the component's own props. Resolved components
// carry concrete trace data (markers are already resolved to literals), so every
// prop is validated strictly against its schema, and a component type absent from
// the catalog allowlist is rejected outright.
const validateComponentProps = (component: Record<string, unknown>): string | undefined => {
  const componentName = component['component'];
  if (typeof componentName !== 'string') {
    return 'A component is missing its string "component" type.';
  }
  if (component['id'] === undefined || component['id'] === null || component['id'] === '') {
    return `Component "${componentName}" is missing a non-empty "id".`;
  }
  const { id: _id, component: _component, ...props } = component;

  const schema = CATALOG_COMPONENT_SCHEMAS[componentName];
  if (!schema) {
    // Closed allowlist: an unknown/injected component type must never reach the
    // renderer (which does not re-check the catalog), so reject it here.
    return `Component "${String(component['id'])}" has an unknown component type "${componentName}". Only components defined in the custom-view catalog are allowed.`;
  }
  const result = schema.safeParse(props);
  if (!result.success) {
    const detail = result.error.issues
      .map((issue) => `${issue.path.join('.') || '(root)'}: ${issue.message}`)
      .join('; ');
    return `Component "${String(component['id'])}" (${componentName}) has invalid props: ${detail}`;
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
      version: A2UI_VERSION,
      createSurface: { surfaceId, catalogId, sendDataModel: true },
    },
  ];

  let sawRoot = false;

  for (const message of rawMessages) {
    if (!isRecord(message)) {
      return { ok: false, error: 'Encountered a message that is not a JSON object.' };
    }

    if ('createSurface' in message || 'deleteSurface' in message) {
      continue;
    }

    if ('updateComponents' in message) {
      let payload: Record<string, unknown> | undefined = isRecord(message['updateComponents'])
        ? { ...message['updateComponents'], surfaceId }
        : undefined;
      // Flatten any nested `props` objects so validation and rendering both see
      // the flat shape we expect.
      if (payload && Array.isArray(payload['components'])) {
        payload = {
          ...payload,
          components: payload['components'].map((component: unknown) =>
            isRecord(component) ? flattenComponentProps(component) : component,
          ),
        };
      }
      const normalized = { version: A2UI_VERSION, updateComponents: payload };
      const parsed = UpdateComponentsMessageSchema.safeParse(normalized);
      if (!parsed.success) {
        return {
          ok: false,
          error: `Invalid updateComponents message: ${parsed.error.issues.map((i) => i.message).join('; ')}`,
        };
      }
      const rawComponents = payload?.['components'];
      const components = Array.isArray(rawComponents) ? rawComponents : [];
      for (const component of components) {
        if (!isRecord(component)) {
          return { ok: false, error: 'A component entry is not a JSON object.' };
        }
        if (component['id'] === 'root') {
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
      const payload = isRecord(message['updateDataModel']) ? { ...message['updateDataModel'], surfaceId } : undefined;
      const normalized = { version: A2UI_VERSION, updateDataModel: payload };
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

    // Unknown / unsupported message shape - ignore it rather than fail the whole
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

// Placeholder surface id stamped on a stored template. The real, host-owned
// surface id is injected later by `validateAndPrepareMessages` when the resolved
// per-trace messages are prepared.
const TEMPLATE_SURFACE_ID = 'main';

// Validates the binding markers + narrative rules for a single template
// component. Unlike `validateComponentProps`, this does NOT strict-check props
// against the catalog schema, because data-bearing props hold `$source` /
// `$spanRef` markers at template time; strict validation happens on the resolved
// per-trace output. Returns an error string, or undefined when the component is
// a valid template component.
const validateTemplateComponent = (component: Record<string, unknown>): string | undefined => {
  const componentName = typeof component['component'] === 'string' ? component['component'] : '(unknown)';
  const id = component['id'] === undefined ? '(no id)' : String(component['id']);

  // Reject an off-catalog component at save time (fail fast). Unlike the per-trace
  // validator we can't strict-check props here — data props still hold markers —
  // but the component TYPE is a fixed literal, so the closed allowlist applies.
  if (!(componentName in CATALOG_COMPONENT_SCHEMAS)) {
    return `Component "${id}" has an unknown component type "${componentName}". Only components defined in the custom-view catalog are allowed.`;
  }

  // A2UI types `id` as optional, but an id-less component is unreferenceable and
  // the per-trace validator rejects it — so without this guard the template saves
  // and then fails to render forever. Fail at save time, while the agent can still
  // repair it.
  if (component['id'] === undefined || component['id'] === null || component['id'] === '') {
    return `Component "${componentName}" is missing a non-empty "id".`;
  }

  if ('renderIfSpan' in component && !isValidSpanRefSelector(unwrapSpanRefSelector(component['renderIfSpan']))) {
    return (
      `Component "${id}" (${componentName}) has an invalid "renderIfSpan" guard. Use a spanRef selector: ` +
      `"root", { "type": "<SPAN_TYPE>", "nth"?: n }, or { "name": "<span name>" }.`
    );
  }

  // The structural "assessments" source materializes into a `string[]` of
  // generated child ids, so it only makes sense on a `children` prop
  // (AssessmentBoard). `resolveComponent` rewrites ANY prop carrying such a
  // marker to that child-id list, so binding one to a scalar prop (`child`,
  // `value`) would silently produce malformed resolved
  // A2UI at per-trace render. Conversely, a non-structural source on `children`
  // resolves to a string/array, not child ids — reject both mismatches at save time.
  for (const [key, propValue] of Object.entries(component)) {
    if (key !== 'children' && isSourceMarker(propValue) && isStructuralSource(propValue.$source)) {
      return (
        `Component "${id}" (${componentName}) binds the structural "${propValue.$source}" source to "${key}", but ` +
        `structural sources may only bind a "children" prop (AssessmentBoard). Use a scalar or "spanField" ` +
        `source for data props.`
      );
    }
    if (key === 'children' && isSourceMarker(propValue) && !isStructuralSource(propValue.$source)) {
      return (
        `Component "${id}" (${componentName}) binds a non-structural "${propValue.$source}" source to "children", but ` +
        `only the structural "assessments" source may bind "children" (AssessmentBoard). Use ` +
        `literal child ids for Row/Column layout, or a scalar or "spanField" source for data props.`
      );
    }
  }

  let error: string | undefined;

  const walk = (value: unknown) => {
    if (error) {
      return;
    }
    if (isSourceMarker(value)) {
      if (!isKnownSource(value.$source)) {
        error = `Component "${id}" (${componentName}) references unknown $source "${value.$source}".`;
        return;
      }
      if (value.$source === SPAN_FIELD_SOURCE_NAME && !isValidSpanFieldMarker(value)) {
        error =
          `Component "${id}" (${componentName}) has an invalid spanField marker. Provide a valid "spanRef" ` +
          `("root" / { "type": "<SPAN_TYPE>", "nth"?: n } / { "name": "<span name>" }) and a "field" of ` +
          `inputs|outputs|attributes|name|spanId. An optional "path" (non-empty array of string keys / number ` +
          `indices, e.g. ["choices", 0, "message", "content"]) is allowed only on inputs|outputs|attributes.`;
        return;
      }
      // Fall through to recurse into the marker's other fields so the narrative
      // rules below apply to any nested strings too.
    } else if (isSpanRefMarker(value)) {
      if (!isValidSpanRefSelector(value.$spanRef)) {
        error =
          `Component "${id}" (${componentName}) has an invalid $spanRef selector. Use "root", ` +
          `{ "type": "<SPAN_TYPE>", "nth"?: n }, or { "name": "<span name>" }.`;
      }
      return;
    }
    if (typeof value === 'string') {
      // Trace-specific narrative is forbidden in a reusable view: a baked
      // `#span:<id>` deeplink points at a span that only exists in the authoring
      // trace, so it breaks on every other trace.
      if (value.includes('#span:')) {
        error =
          `Component "${id}" (${componentName}) contains a "#span:" deeplink. Reusable views cannot embed ` +
          `trace-specific narrative; bind a "spanField" source to select a span by role instead.`;
      }
      return;
    }
    if (Array.isArray(value)) {
      value.forEach(walk);
      return;
    }
    if (isRecord(value)) {
      Object.values(value).forEach(walk);
    }
  };

  walk(component);
  return error;
};

// The A2UI envelope types `updateDataModel.value` as `z.any()`, so the schema
// check above it proves nothing about the contents. A template's data model is
// then persisted verbatim — `resolveTemplate` only swaps markers on component
// props — yet every DynamicString prop may bind to it via `{ "path": ... }`.
// Anything parked here therefore reaches the renderer unresolved and unchecked,
// so apply the same trace-agnostic rules the component walk enforces.
//
// Markers are rejected outright (not merely validated): even a well-formed one
// is never resolved in the data model, so it would render as raw JSON.
const validateTemplateDataModelValue = (value: unknown): string | undefined => {
  let error: string | undefined;

  const walk = (current: unknown) => {
    if (error) {
      return;
    }
    if (isSourceMarker(current) || isSpanRefMarker(current)) {
      const marker = isSourceMarker(current) ? '$source' : '$spanRef';
      error =
        `The template's data model contains a "${marker}" marker. Markers are only resolved on component ` +
        `props, so one placed in the data model renders as raw JSON. Bind the marker to the prop that ` +
        `displays it instead.`;
      return;
    }
    if (typeof current === 'string') {
      if (current.includes('#span:')) {
        error =
          `The template's data model contains a "#span:" deeplink. Reusable views cannot embed ` +
          `trace-specific narrative; bind a "spanField" source to select a span by role instead.`;
      }
      return;
    }
    if (Array.isArray(current)) {
      current.forEach(walk);
      return;
    }
    if (isRecord(current)) {
      Object.values(current).forEach(walk);
    }
  };

  walk(value);
  return error;
};

export type TemplateValidateResult = { ok: true; messages: A2uiMessage[] } | { ok: false; error: string };

/**
 * Validates a trace-agnostic custom view TEMPLATE (authored once by the LLM with
 * `$source` / `$spanRef` markers). Unlike `validateAndPrepareMessages`, this is
 * marker-aware and lenient on data props: it
 *
 *  - extracts the message array and drops any createSurface/deleteSurface,
 *  - normalizes the surface id to a placeholder + flattens nested props,
 *  - validates each envelope (Zod) and each marker (known source name / valid
 *    spanRef selector) and rejects forbidden trace-specific narrative,
 *  - walks any `updateDataModel` value for the same narrative rules, since
 *    components can read it back through a `{ "path": ... }` binding,
 *  - requires a `root` component,
 *
 * returning the marker-preserving template to persist. Per-trace rendering then
 * runs `resolveTemplate` (to swap markers for this trace's data) followed by
 * `validateAndPrepareMessages` (to strict-validate the resolved components).
 */
export const validateTemplate = (raw: unknown): TemplateValidateResult => {
  const rawMessages = toMessageArray(raw);
  if (!rawMessages || rawMessages.length === 0) {
    return { ok: false, error: 'The model did not return any A2UI messages.' };
  }

  const kept: A2uiMessage[] = [];
  let sawRoot = false;

  for (const message of rawMessages) {
    if (!isRecord(message)) {
      return { ok: false, error: 'Encountered a message that is not a JSON object.' };
    }
    if ('createSurface' in message || 'deleteSurface' in message) {
      continue;
    }

    if ('updateComponents' in message) {
      let payload: Record<string, unknown> | undefined = isRecord(message['updateComponents'])
        ? { ...message['updateComponents'], surfaceId: TEMPLATE_SURFACE_ID }
        : undefined;
      if (payload && Array.isArray(payload['components'])) {
        payload = {
          ...payload,
          components: payload['components'].map((component: unknown) =>
            isRecord(component) ? flattenComponentProps(component) : component,
          ),
        };
      }
      const normalized = { version: A2UI_VERSION, updateComponents: payload };
      const parsed = UpdateComponentsMessageSchema.safeParse(normalized);
      if (!parsed.success) {
        return {
          ok: false,
          error: `Invalid updateComponents message: ${parsed.error.issues.map((i) => i.message).join('; ')}`,
        };
      }
      const rawComponents = payload?.['components'];
      const components = Array.isArray(rawComponents) ? rawComponents : [];
      for (const component of components) {
        if (!isRecord(component)) {
          return { ok: false, error: 'A component entry is not a JSON object.' };
        }
        if (component['id'] === 'root') {
          sawRoot = true;
        }
        const componentError = validateTemplateComponent(component);
        if (componentError) {
          return { ok: false, error: componentError };
        }
      }
      kept.push(parsed.data);
      continue;
    }

    if ('updateDataModel' in message) {
      const payload = isRecord(message['updateDataModel'])
        ? { ...message['updateDataModel'], surfaceId: TEMPLATE_SURFACE_ID }
        : undefined;
      const normalized = { version: A2UI_VERSION, updateDataModel: payload };
      const parsed = UpdateDataModelMessageSchema.safeParse(normalized);
      if (!parsed.success) {
        return {
          ok: false,
          error: `Invalid updateDataModel message: ${parsed.error.issues.map((i) => i.message).join('; ')}`,
        };
      }
      const dataModelError = validateTemplateDataModelValue(parsed.data.updateDataModel.value);
      if (dataModelError) {
        return { ok: false, error: dataModelError };
      }
      kept.push(parsed.data);
      continue;
    }
  }

  if (!sawRoot) {
    return { ok: false, error: 'The generated UI has no "root" component to render.' };
  }

  return { ok: true, messages: kept };
};
