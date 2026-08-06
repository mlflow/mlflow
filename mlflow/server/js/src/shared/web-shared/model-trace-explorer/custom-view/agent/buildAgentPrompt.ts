// A2UI authoring building blocks for the custom view. Provides the component
// catalog / data-binding references and examples, the per-trace data snapshot,
// and the MLflow Assistant authoring guide. Follows A2UI v0.9's "prompt-first"
// contract: the schema + examples are embedded in the prompt and the model
// passes the full message stream as the render_custom_view tool's "messages"
// argument, which the host validates before processing.

import type { AgentAssessment } from '../customViewBuilders';
import { A2UI_VERSION } from './validateA2uiMessages';

export const RENDER_CUSTOM_VIEW_TOOL_NAME = 'render_custom_view';

// One span's entry in the nodeMap JSON handed to the model. Keyed by span id,
// this is just the trace's nodeMap serialized to plain JSON (no curated shape).
export type AgentNode = {
  name: string;
  type: string;
  startMs: number;
  endMs: number;
  durationMs: number;
  parentId?: string;
  inputs: unknown;
  outputs: unknown;
};

// The trace data the model can use. `nodeMap` is the raw per-span source
// (including inputs/outputs) the model parses to extract what it needs; the
// other fields are precomputed conveniences for common views.
export type AgentTraceData = {
  metrics: Record<string, unknown>;
  // The trace's nodeMap as plain JSON, keyed by span id. The model parses this
  // and binds the data it needs into components via the A2UI data model.
  nodeMap?: Record<string, AgentNode>;
  // The trace's real assessments (LLM-judge / human feedback).
  assessments?: AgentAssessment[];
};

// The surface id is a fixed placeholder; the host rewrites it to a unique id
// after generation, so the model never has to invent one.
const PLACEHOLDER_SURFACE_ID = 'main';

export const CATALOG_REFERENCE = `Available components (use the "component" field with these exact names). For DATA-bearing props you do NOT write literal trace values — you write a binding MARKER (see "Data binding" below). Layout text (titles, labels, icons, tones) stays a literal.

- "Row": horizontal layout. props: { "children": [<child ids>], "align"?: "start"|"center"|"end"|"stretch" }. For EQUAL-WIDTH columns, give EVERY direct child the SAME "weight" (e.g. "weight": 1) and set "align": "stretch" (so they also match heights). Without "weight" each child shrinks to its own content and the columns look uneven.
- "Column": vertical layout. props: { "children": [<child ids>] }
- "Text": a single-line/short text label rendered as real typography (NOT Markdown — do not put #, *, or _ in "text"; use the "Markdown" component for bold/italic/lists/multi-line). props: { "text": <string OR a spanField source marker>, "variant"?: "h1"|"h2"|"h3"|"h4"|"h5"|"caption"|"body", "weight"?: <number> }. Use "variant" to build hierarchy: a Card/section title should be an "h3" or "h4" heading (larger, bold); small secondary metadata should be "caption" (small + muted). "text" is normally a STATIC label, but to show a SPECIFIC span's name or id as a per-trace caption you may bind it to a "spanField" source, e.g. { "$source": "spanField", "spanRef": { "type": "TOOL", "nth": 0 }, "field": "name" } (see "Data binding"). Use static text for headings, spanField for per-span name/id captions.
- "Card": a bordered container around a SINGLE child. props: { "child": <child id>, "weight"?: <number> }. To put multiple elements in a card, wrap them in a Row/Column and pass that container's id as the child. When cards sit side by side in a Row, give each the SAME "weight" (e.g. "weight": 1) so they share the width equally.
- "Icon": a single Databricks Design System icon. props: { "name": <string>, "size"?: <number> }. Use a camelCase name, e.g. "check", "warning", "error", "info", "search". Use sparingly — most components already carry their own icon prop.
- "StatCard": a single metric tile. props: { "value": <SCALAR SOURCE marker>, "label": <string>, "icon"?: "wrench"|"clock"|"checkCircle"|"xCircle"|"hash"|"checklist", "tone"?: "info"|"success"|"warning"|"danger", "weight"?: <number> }. Bind "value" to a scalar source, e.g. { "$source": "metrics.latency" }. "label", "icon", "tone" are static — authored ONCE, they CANNOT follow the value, so never style a tile as if you knew what its value will be. "info" is the ONLY neutral tone — "success", "warning", and "danger" each assert a verdict, and "checkCircle"/"xCircle" each assert an outcome. Since a StatCard's "value" is always a bound marker, default to "tone": "info" with a descriptive icon ("clock" for latency, "hash" for counts, "checklist" for status): a status tile styled "success"/"checkCircle" still renders a green check on a trace whose status is ERROR, and a latency tile styled "warning" still renders a yellow tint on a fast trace. Reach for a verdict tone ONLY when it holds for EVERY trace because the LABEL itself is the verdict (e.g. an error-count tile). StatCards already flex to equal width in a Row; only set "weight" to bias the split.
- "Markdown": a markdown text block. props: { "text": <static markdown string OR a spanField source marker>, "title"?: <string heading> }. "text" is normally a STATIC instruction/heading, but to render ONE span's value as prose (e.g. the model's answer) you may bind it to a "spanField" source whose "path" lands on a SCALAR leaf, e.g. { "$source": "spanField", "spanRef": { "type": "LLM", "nth": 0 }, "field": "outputs", "path": ["choices", 0, "message", "content"] } (see "Data binding"); a marker that resolves to an object dumps raw JSON, so use "KeyValueViewer" for whole objects. Markdown you write YOURSELF must stay trace-agnostic — no trace-specific narrative and no "#span:" deeplinks (those would point at a span that only exists in one trace and break when the view is reused). Prefer Text for short headings.
- "KeyValueViewer": displays a SINGLE labeled value. props: { "label"?: <string>, "value": <string OR a spanField source marker>, "initialFormat"?: "json"|"text"|"markdown", "hideFormatToggle"?: <bool> }. To show ONE specific span's input/output/attributes inline (e.g. a tool call's output in its own card), bind "value" to a "spanField" source with "initialFormat": "json", e.g. { "$source": "spanField", "spanRef": { "type": "TOOL", "nth": 0 }, "field": "outputs" } (see "Data binding"); the host re-resolves it per trace. To cover SEVERAL spans, author one card per span with its own "spanField" marker and a "renderIfSpan" guard. Do NOT paste literal span JSON into "value".
- "AssessmentCard": a single colored box for one assessment. You normally do NOT emit these directly — bind an AssessmentBoard's children to the assessments source and the host materializes one card per assessment. props (for reference): { "name", "value"?, "rationale"?, "source"?, "sentiment"? }.
- "AssessmentBoard": a wrapping container for assessment cards. props: { "title"?: <string>, "icon"?: "checklist"|"list"|"checkCircle", "children": <STRUCTURAL SOURCE marker>, "emptyMessage"?: <string> }. For ANY request about judge results / evaluations / assessments, bind "children" to { "$source": "assessments" }; the host builds one AssessmentCard per real assessment in the current trace. Do NOT hand-author AssessmentCards.`;

export const BINDING_REFERENCE = `Data binding (CRITICAL — author ONCE, reused for every trace):
You are authoring a REUSABLE, trace-agnostic TEMPLATE, not a one-off view. The "traceSample" in the context is ONLY an example of the shape of the data; you must NOT copy its concrete values, span ids, rows, or counts into the spec. Instead, every data-bearing prop is a binding MARKER that the host re-resolves against whatever trace is open. The marker kind is "$source" (data props), over a CLOSED set of sources, plus BARE span selectors used by "spanField" and "renderIfSpan":

1. "$source" markers — fill a data prop with the current trace's data. Write { "$source": "<name>" } (structural sources also accept extra fields). Available sources:
   - Scalars (for a StatCard "value" or any scalar data prop): "metrics.status", "metrics.latency", "metrics.totalTokens", "metrics.assessments" (the latter is the COUNT of assessments).
   - Structural (for a "children" prop): "assessments" (AssessmentBoard — one AssessmentCard per real assessment).
   - Per-span field (for a KeyValueViewer "value" or a Text "text", to show ONE specific span's data inline): "spanField". Write { "$source": "spanField", "spanRef": <selector>, "field": "<field>" } where "spanRef" is a BARE selector — exactly "root", { "type": "<SPAN_TYPE>", "nth"?: n }, or { "name": "<span name>" } — and "field" is one of "inputs", "outputs", "attributes" (rendered as JSON — use "initialFormat": "json"), "name", or "spanId" (short text). Write "spanRef" as the bare selector — "spanRef": "root" or "spanRef": { "type": "TOOL", "nth": 0 } — not wrapped in any marker object. Use this for "show the output of the first tool call"-style requests: bind the card's KeyValueViewer to { "$source": "spanField", "spanRef": { "type": "TOOL", "nth": 0 }, "field": "outputs" }, and the trace's user prompt to { "$source": "spanField", "spanRef": "root", "field": "inputs" }. If the selected span is absent in a trace, JSON fields resolve to an empty/null value and text fields to "" (an empty card), so the view degrades gracefully instead of showing another trace's data.
     - Nested "path" (extract ONE nested value out of "inputs"/"outputs"/"attributes"): add an OPTIONAL "path" — an array of string keys and/or number indices drilling into that field's JSON — e.g. { "$source": "spanField", "spanRef": { "type": "LLM", "nth": 0 }, "field": "outputs", "path": ["choices", 0, "message", "content"] } or { ..., "field": "outputs", "path": ["content"] }. When the leaf is a SCALAR (string/number/bool) it renders as readable prose, so put it in a "Text" (or a "Markdown"), or a "KeyValueViewer" WITHOUT "initialFormat": "json" — this is how you turn a model's response into standalone text instead of a JSON blob. When the leaf is itself an object/array it still renders as JSON (keep "initialFormat": "json"). Prefer extracting the specific nested field the user asked for (e.g. the prompt text, the model's answer) over dumping the whole object. "path" is a STRUCTURAL, trace-agnostic selector into the field's SCHEMA — NOT a value copied from the traceSample: pick keys that are stable across traces, treat number indices as positional (like "nth"), and never bake sample-specific structure in. If the path is missing in some trace, it resolves to "" so the view still degrades gracefully. "path" applies ONLY to "inputs"/"outputs"/"attributes" (not "name"/"spanId").

Conditional rendering:
- Any component may carry a "renderIfSpan": <bare selector> guard ("root" / { "type": "<SPAN_TYPE>", "nth"?: n } / { "name": "<span name>" }). The host OMITS that component AND its entire subtree when the selector matches no span in the current trace. ALWAYS put a "renderIfSpan" on each per-span card (e.g. the nth tool-call card) so a fixed N-card layout collapses to only the cards whose span exists — otherwise the extra card renders an empty/"null" output. Use the SAME selector as the card's spanField. (Like spanField's "spanRef", this is a BARE selector.)

Rules:
- Do NOT inline literal trace data anywhere a marker belongs. A StatCard value is { "$source": "metrics.latency" }, NOT "1.20s". A specific span's output in a KeyValueViewer is a "spanField" marker, NOT pasted JSON. A span id/name caption is a "spanField" marker, NOT a literal id copied from the traceSample.
- Only these source names exist. If the user asks for data with no matching source, either bind a "spanField" (for one span's input/output/attributes/name/id) or state it is unavailable — do NOT invent a source name.
- Static layout text (titles, labels, icons, tones) stays a plain literal — it is the SAME for every trace. Because it cannot change, it must not imply anything about a value that does: give a per-trace status a neutral icon/tone rather than a hard-coded success or danger.`;

export const LAYOUT_GUIDANCE = `Layout & visual polish (make views look designed, not flat):
1. Give structure with Cards. Group each logical section (a span, a response, a metrics summary) into its own "Card". A Card holds ONE child, so put a "Column" inside it.
2. Title every Card. Make the FIRST child of a Card's Column a heading — a "Text" with "variant": "h3" or "h4" (or "h5" for sub-sections). Don't leave a bare body-text line acting as the title.
3. Demote metadata to captions. Render span ids, durations, timestamps, and other secondary details as a "Text" with "variant": "caption" (small + muted), e.g. { "text": "span: <id>", "variant": "caption" } — not as plain body text.
4. Use tiles for metrics. For single numbers/statuses (latency, tokens, status, counts) prefer "StatCard"s in a "Row" with "align": "stretch" rather than text lines.
5. Lay out intentionally and keep columns EVEN. Put side-by-side cards in a "Row" with "align": "stretch" AND give every card in that Row the SAME "weight" (e.g. "weight": 1) so they share the width equally and match heights — without equal "weight" the cards shrink to their content and look uneven. Use a "Column" for stacked sections. When you render one card PER repeated item (e.g. one Card per tool call), lay them out with a CONSISTENT number of equal-weight cards per Row (e.g. 2 per Row). If the items don't divide evenly and the final Row is left with a SINGLE card, let that card span the FULL width — put it alone in a "Row" with "weight": 1 (or place it directly in the parent "Column") so it covers the whole row instead of sitting half-width next to empty space. Do NOT add empty spacer columns.
6. Emphasize with Markdown. For bold/italic labels, lists, or multi-line prose use the "Markdown" component (the "Text" component is plain — never put # or ** inside its "text").
7. Aim for a consistent rhythm: heading -> optional caption -> content, inside each Card.`;

export const OUTPUT_RULES = `Output format rules (A2UI ${A2UI_VERSION}):
1. Deliver the view by CALLING the \`${RENDER_CUSTOM_VIEW_TOOL_NAME}\` tool with two arguments — "title" (a SHORT 2-5 word human-readable name describing the view, e.g. "Trace Summary", "Span Cards", "Agent Key Actions" — NOT the user's raw prompt) and "messages" (the A2UI message array described by all rules below). Do NOT print the JSON in the chat or a code fence — only the tool call updates the view. (The examples below show JSON for illustration of the "messages" content; pass that array as the "messages" argument, not as a chat message.)
2. EVERY message object MUST include "version": "${A2UI_VERSION}".
3. Each message object contains EXACTLY ONE of: "createSurface", "updateComponents", "updateDataModel".
4. Do NOT emit "createSurface" or "deleteSurface" — the host creates the surface for you. Emit only "updateComponents" (and optionally "updateDataModel"). Always use "surfaceId": "${PLACEHOLDER_SURFACE_ID}".
5. The "updateComponents" message has { "surfaceId": "${PLACEHOLDER_SURFACE_ID}", "components": [...] }.
6. Components are a flat adjacency list: each has a unique "id" and a "component" type. Reference children by their string ids in a "children" array (do NOT nest component objects). Put EVERY prop DIRECTLY on the component object alongside "id" and "component" — do NOT wrap them in a "props" object. Write { "id": "c1", "component": "Card", "child": "c2" }, NOT { "id": "c1", "component": "Card", "props": { "child": "c2" } }.
7. There MUST be exactly one component with "id": "root", and it MUST be the first component. Parents must appear before their children.
8. DATA props are binding MARKERS, never literal trace data (see "Data binding"). A "children" prop that shows trace data (AssessmentBoard) is an { "$source": "assessments" } marker; a scalar data prop (StatCard "value") is an { "$source": "<scalar source>" } marker. Layout-only arrays (a Row/Column "children" listing your OWN component ids) stay literal.
9. SCALAR data props are { "$source": "metrics.*" } markers — do NOT inline a literal like "1.83s" copied from the traceSample. The traceSample shows example shapes only; its concrete values belong to one trace and must NOT appear in your spec.
10. Trace-specific narrative is FORBIDDEN. Do NOT write prose that describes what a specific trace did, and do NOT emit "#span:" deeplinks — the saved view is reused across many traces, so such text would be wrong for every other trace. Markdown you write by hand is STATIC instructions/headings only; per-trace content must come from a binding marker, never from hand-written prose.
11. Only use the component types, props, source names, and spanRef selectors listed above. Do not invent components, props, icon names, enum values, or source names.
12. CRITICAL — never fabricate data. The only trace data that appears is what the bound sources resolve to; do NOT invent metrics, scores, counts, failure patterns, or recommendations. This is ONE single trace, so never reference cross-trace aggregates. The ONLY judge/evaluation results are the "assessments" source; "metrics.assessments" is merely their COUNT.
13. If the user asks for data that has no matching source, do NOT make one up. Bind a "spanField" marker (for one span's input/output/attributes/name/id), or render a single short static message stating it is unavailable (e.g. a "Text" with "text": "Not available in this view."). It is better to say the data is unavailable than to fabricate or hardcode values.
14. NEVER paste a specific span's input/output JSON as a literal "value", and NEVER copy a span id/name from the traceSample into "text" or a spanRef selector. Per-span data and ids are ALWAYS "spanField" markers so they re-resolve for every trace; a literal would freeze the view to the authoring trace.`;

export const EXAMPLE: string = `Example — a heading plus a metrics row. Note: the StatCard "value"s are binding MARKERS (no literal trace data); only the layout, heading, and labels are literal (and note the { "title", "messages" } wrapper):
\`\`\`json
{
  "title": "Trace Summary",
  "messages": [
    {
      "version": "v0.9",
      "updateComponents": {
        "surfaceId": "${PLACEHOLDER_SURFACE_ID}",
        "components": [
          { "id": "root", "component": "Column", "children": ["heading", "metrics"] },
          { "id": "heading", "component": "Text", "text": "Trace Summary", "variant": "h4" },
          { "id": "metrics", "component": "Row", "children": ["stat-latency", "stat-tokens", "stat-status"], "align": "stretch" },
          { "id": "stat-latency", "component": "StatCard", "value": { "$source": "metrics.latency" }, "label": "Latency", "icon": "clock", "tone": "info" },
          { "id": "stat-tokens", "component": "StatCard", "value": { "$source": "metrics.totalTokens" }, "label": "Total Tokens", "icon": "hash", "tone": "info" },
          { "id": "stat-status", "component": "StatCard", "value": { "$source": "metrics.status" }, "label": "Status", "icon": "checklist", "tone": "info" }
        ]
      }
    }
  ]
}
\`\`\``;

export const CARD_STYLE_EXAMPLE: string = `Example — a DECORATED card (apply this styling pattern to every card: titled heading, caption, then bound content). Here a summary card shows bound metrics and the root span's input via a "spanField" marker (note there is NO literal span id or per-trace value):
\`\`\`json
[
  {
    "version": "v0.9",
    "updateComponents": {
      "surfaceId": "${PLACEHOLDER_SURFACE_ID}",
      "components": [
        { "id": "root", "component": "Column", "children": ["summary-card"] },
        { "id": "summary-card", "component": "Card", "child": "summary-col" },
        { "id": "summary-col", "component": "Column", "children": ["summary-title", "summary-caption", "metrics-row", "prompt"] },
        { "id": "summary-title", "component": "Text", "text": "Trace summary", "variant": "h4" },
        { "id": "summary-caption", "component": "Text", "text": "Status and latency for this trace", "variant": "caption" },
        { "id": "metrics-row", "component": "Row", "align": "stretch", "children": ["stat-status", "stat-latency"] },
        { "id": "stat-status", "component": "StatCard", "value": { "$source": "metrics.status" }, "label": "Status", "icon": "checklist", "tone": "info", "weight": 1 },
        { "id": "stat-latency", "component": "StatCard", "value": { "$source": "metrics.latency" }, "label": "Latency", "icon": "clock", "tone": "info", "weight": 1 },
        { "id": "prompt", "component": "KeyValueViewer", "label": "User prompt", "value": { "$source": "spanField", "spanRef": "root", "field": "inputs" }, "initialFormat": "json" }
      ]
    }
  }
]
\`\`\`
Note the heading ("variant":"h4") as the card title, the caption line, the bound StatCard "value"s, and the root span's input bound via a "spanField" marker — never a literal value or span id.`;

export const SPAN_CARD_EXAMPLE: string = `Example — per-span cards that show the OUTPUT of specific tool calls side by side (the "show the first N tool calls" pattern). The key idea: select each span by ROLE with "spanRef": { "type": "TOOL", "nth": <n> } (NOT by a baked span id/name), and bind its caption and output to "spanField" markers. The card COUNT is fixed by the layout you author (here 2). To avoid a phantom empty card on a trace with FEWER tool calls, put a "renderIfSpan": <bare selector> guard on EACH per-span card (matching that card's span): the host OMITS the card and everything inside it when that span is absent in the current trace — so a 1-tool trace shows just one card, with NO null output:
\`\`\`json
[
  {
    "version": "v0.9",
    "updateComponents": {
      "surfaceId": "${PLACEHOLDER_SURFACE_ID}",
      "components": [
        { "id": "root", "component": "Column", "children": ["header", "cards"] },

        { "id": "header", "component": "Card", "child": "header-col" },
        { "id": "header-col", "component": "Column", "children": ["header-title", "header-prompt"] },
        { "id": "header-title", "component": "Text", "text": "User prompt", "variant": "h3" },
        { "id": "header-prompt", "component": "KeyValueViewer", "label": "Question", "value": { "$source": "spanField", "spanRef": "root", "field": "inputs" }, "initialFormat": "json" },

        { "id": "cards", "component": "Row", "align": "stretch", "children": ["card-0", "card-1"] },

        { "id": "card-0", "component": "Card", "child": "col-0", "weight": 1, "renderIfSpan": { "type": "TOOL", "nth": 0 } },
        { "id": "col-0", "component": "Column", "children": ["name-0", "id-0", "out-0"] },
        { "id": "name-0", "component": "Text", "text": { "$source": "spanField", "spanRef": { "type": "TOOL", "nth": 0 }, "field": "name" }, "variant": "h4" },
        { "id": "id-0", "component": "Text", "text": { "$source": "spanField", "spanRef": { "type": "TOOL", "nth": 0 }, "field": "spanId" }, "variant": "caption" },
        { "id": "out-0", "component": "KeyValueViewer", "label": "Output", "value": { "$source": "spanField", "spanRef": { "type": "TOOL", "nth": 0 }, "field": "outputs" }, "initialFormat": "json" },

        { "id": "card-1", "component": "Card", "child": "col-1", "weight": 1, "renderIfSpan": { "type": "TOOL", "nth": 1 } },
        { "id": "col-1", "component": "Column", "children": ["name-1", "id-1", "out-1"] },
        { "id": "name-1", "component": "Text", "text": { "$source": "spanField", "spanRef": { "type": "TOOL", "nth": 1 }, "field": "name" }, "variant": "h4" },
        { "id": "id-1", "component": "Text", "text": { "$source": "spanField", "spanRef": { "type": "TOOL", "nth": 1 }, "field": "spanId" }, "variant": "caption" },
        { "id": "out-1", "component": "KeyValueViewer", "label": "Output", "value": { "$source": "spanField", "spanRef": { "type": "TOOL", "nth": 1 }, "field": "outputs" }, "initialFormat": "json" }
      ]
    }
  }
]
\`\`\``;

// Max spans surfaced in the prompt snapshot (applied to the raw nodeMap entries)
// so a large trace can't blow up the context.
const MAX_PROMPT_SPANS = 400;

// Keeps small inputs/outputs structured, but truncates large payloads to a
// string so a single span can't blow up the prompt.
const truncateValue = (value: unknown, max = 2000): unknown => {
  if (value === null || value === undefined) {
    return value;
  }
  const str = JSON.stringify(value);
  if (str === undefined) {
    return value;
  }
  return str.length <= max ? value : `${str.slice(0, max)}… (truncated)`;
};

// Builds the compact, capped/truncated trace snapshot embedded in the prompt.
// This is the model's source of truth for the CURRENT trace's data SHAPE: it
// shows the available metrics/spans/assessments so the model can
// pick sensible sources, but the authored template binds markers (never these
// concrete values). Caps/truncation keep a large trace from blowing up the
// prompt (very large values may be truncated in the rendered UI).
export const buildAgentDataSnapshot = (data: AgentTraceData): Record<string, unknown> => {
  // Serialize the nodeMap for the prompt: cap the number of spans and truncate
  // each span's inputs/outputs so a large trace can't blow up the context.
  const nodeMapEntries = Object.entries(data.nodeMap ?? {});
  const cappedEntries = nodeMapEntries.slice(0, MAX_PROMPT_SPANS);
  const nodeMapJson = Object.fromEntries(
    cappedEntries.map(([id, node]) => [
      id,
      { ...node, inputs: truncateValue(node.inputs), outputs: truncateValue(node.outputs) },
    ]),
  );
  const nodeMapTruncated = Math.max(nodeMapEntries.length - cappedEntries.length, 0);

  return {
    metrics: data.metrics,
    // Raw per-span source (including inputs/outputs), keyed by span id.
    nodeMap: nodeMapJson,
    nodeMapTruncated,
    // The trace's real assessments (LLM-judge / human feedback). This is the
    // ONLY evaluation/judge data available.
    assessments: data.assessments ?? [],
  };
};

// Static authoring guide handed to MLflow Assistant via page context (the
// assistant has no built-in knowledge of A2UI or the custom view). Delivery is
// via a real `render_custom_view` tool call — see supports_client_tools on the
// backend provider, which gates whether this guide is published at all. CLI
// providers (Claude Code, Codex) have no mid-stream client-tool channel and are
// not supported yet (see the fenced-block-convention follow-up in the plan).
export const buildCustomViewAuthoringGuide = (): string => {
  const intro = [
    'CUSTOM TRACE VIEW AUTHORING MODE.',
    'The user is viewing the "Custom View" tab of the MLflow trace explorer and wants you to BUILD or MODIFY an A2UI view ("custom view"). You author a REUSABLE, TRACE-AGNOSTIC TEMPLATE exactly ONCE: the host saves it and re-binds it to every trace the user cycles through WITHOUT calling you again. So you must NOT bake in the current trace\'s data. The "traceSample" in the context shows only the SHAPE of the data (so you can pick sensible sources and layout); never copy its concrete span ids, values, rows, or counts into the spec. Every data-bearing prop must be a binding MARKER (see the Data binding rules) that the host resolves per trace.',
    `When (and only when) the user asks you to build, change, or update the custom view, CALL the \`${RENDER_CUSTOM_VIEW_TOOL_NAME}\` tool. Pass two arguments: "title" (a short 2-5 word view name, e.g. "Trace Summary", "Span Cards" — NOT the user's raw words) and "messages" (the FULL A2UI message stream as a JSON array). Calling the tool is the ONLY way the view updates — do NOT print the JSON spec as a chat message or in a code fence, because nothing applies a spec that is merely written in chat (the view will silently stay unchanged). After the tool call succeeds, reply with a short natural-language confirmation of what changed. If the user is just asking a question (not requesting a view change), answer normally without calling the tool.`,
    'Always pass the COMPLETE "messages" spec for the single view (not a diff) on every turn that updates it. When a "currentTemplate" is provided in the context, it is the existing reusable template (with its binding markers): KEEP its layout, component choices, and markers, and apply ONLY the change the user asked for. Do NOT replace markers with literal data and do NOT introduce trace-specific values. Re-pick the "title" whenever an edit changes what the view shows; keep the previous title for purely cosmetic edits (colors, spacing, minor wording).',
  ].join('\n');

  return [
    intro,
    CATALOG_REFERENCE,
    BINDING_REFERENCE,
    LAYOUT_GUIDANCE,
    OUTPUT_RULES,
    EXAMPLE,
    CARD_STYLE_EXAMPLE,
    SPAN_CARD_EXAMPLE,
  ].join('\n\n');
};
