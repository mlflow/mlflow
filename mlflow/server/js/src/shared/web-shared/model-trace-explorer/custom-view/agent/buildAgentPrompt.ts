// Builds the chat messages sent to the gateway LLM to generate an A2UI message
// stream for one custom-view dashboard block. Follows A2UI v0.9's "prompt-first"
// contract: the schema + examples are embedded in the prompt and the model
// returns the full message stream, which the host validates before processing.

export type AgentChatMessage = { role: 'system' | 'user'; content: string };

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

// A real assessment on the trace (LLM-judge or human feedback / expectation).
// This is the actual evaluation data — the model must use these values rather
// than inventing scores or judge results.
export type AgentAssessment = {
  name: string;
  value: unknown;
  rationale?: string;
  source: string;
  spanId?: string;
  error?: string;
};

// The trace data the model can use. `nodeMap` is the raw per-span source
// (including inputs/outputs) the model parses to extract what it needs; the
// other fields are precomputed conveniences for common views.
export type AgentTraceData = {
  metrics: Record<string, unknown>;
  toolRows: { color?: string; cells: string[] }[];
  timelineRows: { label: string; start: number; end: number; depth: number }[];
  treeNodes: unknown[];
  // The trace's nodeMap as plain JSON, keyed by span id. The model parses this
  // and binds the data it needs into components via the A2UI data model.
  nodeMap?: Record<string, AgentNode>;
  // The trace's real assessments (LLM-judge / human feedback). This is the ONLY
  // source of evaluation/judge results — there are no other scores.
  assessments?: AgentAssessment[];
};

// The surface id is a fixed placeholder; the host rewrites it to a unique id
// after generation, so the model never has to invent one.
const PLACEHOLDER_SURFACE_ID = 'main';

const CATALOG_REFERENCE = `Available components (use the "component" field with these exact names):

- "Row": horizontal layout. props: { "children": [<child ids>], "align"?: "start"|"center"|"end"|"stretch" }
- "Column": vertical layout. props: { "children": [<child ids>] }
- "Text": plain text. props: { "text": <string> }
- "StatCard": a single metric tile. props: { "value": <string>, "label": <string>, "icon"?: "wrench"|"clock"|"checkCircle"|"xCircle"|"hash"|"checklist", "tone"?: "info"|"success"|"warning"|"danger" }
- "DataTable": a column-aligned table. props: { "title"?: <string>, "icon"?: "list"|"wrench"|"clock"|"hash"|"checklist", "columns": [{ "label": <string>, "align"?: "left"|"center"|"right" }], "rows": [{ "color"?: <css color>, "cells": [<string>, ...] }], "emptyMessage"?: <string> }. Each row's "cells" are positional, aligned to "columns" by index.
- "TimelineChart": a Gantt-style timeline. props: { "title"?: <string>, "icon"?: "list"|"wrench"|"clock"|"hash"|"checklist", "rows": [{ "label": <string>, "start": <number ms>, "end": <number ms>, "depth"?: <number>, "color"?: <css color> }], "emptyMessage"?: <string> }
- "TreeView": a collapsible span tree. props: { "title"?: <string>, "nodes": [<tree node>], "emptyMessage"?: <string> }. A tree node is { "id": <string>, "label": <string>, "icon": <string>, "hasException": <bool>, "isRootSpan": <bool>, "badge"?: <string>, "attributes": {...}, "children": [<tree node>, ...] }.
- "ContentViewer": a titled key/value content block. props: { "title"?: <string>, "icon"?: "list"|"wrench"|"clock"|"hash"|"checklist", "fields": [{ "label"?: <string>, "value": <JSON-encoded string> }], "collapsible"?: <bool>, "emptyMessage"?: <string> }
- "AssessmentCard": a single colored box for one assessment/judge result. props: { "name": <string>, "value"?: <string>, "rationale"?: <string>, "source"?: <string>, "sentiment"?: "positive"|"negative"|"neutral"|"error" }. Set "name" to the assessment name, "value" to a SHORT verdict (e.g. "yes"/"no"/"Error" — never a long string), "rationale" to its rationale (put any long error message here, not in "value"), "source" to its source, and "sentiment" to "positive" for yes/true/pass values, "negative" for no/false/fail values, "error" if it has an error, else "neutral".
- "AssessmentBoard": a wrapping container for AssessmentCards. props: { "title"?: <string>, "icon"?: "checklist"|"list"|"checkCircle", "children": [<AssessmentCard ids>], "emptyMessage"?: <string> }. For any request about judge results / evaluations / feedback, emit one AssessmentCard per entry in the "assessments" data and list their ids in this board's "children".`;

const OUTPUT_RULES = `Output format rules (A2UI v0.9):
1. Respond with ONLY a single JSON array of message objects, wrapped in a \`\`\`json code fence. No prose before or after.
2. EVERY message object MUST include "version": "v0.9".
3. Each message object contains EXACTLY ONE of: "createSurface", "updateComponents", "updateDataModel".
4. Do NOT emit "createSurface" or "deleteSurface" — the host creates the surface for you. Emit only "updateComponents" (and optionally "updateDataModel"). Always use "surfaceId": "${PLACEHOLDER_SURFACE_ID}".
5. The "updateComponents" message has { "surfaceId": "${PLACEHOLDER_SURFACE_ID}", "components": [...] }.
6. Components are a flat adjacency list: each has a unique "id" and a "component" type. Reference children by their string ids in a "children" array (do NOT nest component objects).
7. There MUST be exactly one component with "id": "root", and it MUST be the first component. Parents must appear before their children.
8. ARRAY-VALUED props MUST be literal arrays — you CANNOT bind them with { "path": ... }. This applies to: "nodes" (TreeView), "rows" and "columns" (DataTable/TimelineChart), "fields" (ContentViewer), and a row's "cells". Inline these arrays directly from the provided data. For a tree of ALL spans, set TreeView "nodes" to the provided "treeNodes" array verbatim; but if the request is for a SUBSET (e.g. only tool calls), inline only the matching nodes (filtered by span type) instead of the whole array — see the filtering note in the data section.
9. Only SCALAR string props may use { "path": "/..." } bindings: a StatCard "value"/"label", a "Text" "text", a single DataTable cell value, or a single ContentViewer field "value". To use a binding, emit ONE "updateDataModel" message whose "value" object holds the extracted data, then reference it by JSON pointer. When a span input/output is an object you bind into a cell or Text, JSON-stringify it first so it renders as text. If you are unsure, prefer inlining literal values rather than binding.
10. Only use the component types and props listed in the catalog. Do not invent components, props, icon names, or enum values.
11. CRITICAL — never fabricate data. Use ONLY values that appear literally in the provided trace data. Do NOT invent, estimate, or infer metrics, scores, counts, percentages, failure patterns, recommendations, or config values that are not present. In particular: this is ONE single trace (not a corpus), so never reference a number of "traces analyzed"/"low-score traces" or any cross-trace aggregate. The ONLY judge/evaluation results are the entries in "assessments" (each with name/value/rationale/source); "metrics.assessments" is merely their COUNT. There are NO retrieval scores, average scores, failure patterns, threshold/chunk-size settings, or config recommendations unless they appear verbatim in "assessments" or a span's inputs/outputs.
12. If the requested information is not present in the provided data, do NOT make something up. Instead render a single short message stating it is unavailable (e.g. a "Text" component with "text": "No LLM judge feedback is available in this trace." as the root, or a "StatCard" with value "N/A"). It is better to say the data is unavailable than to display fabricated values.`;

const EXAMPLE = `Example — a tool performance table plus a metrics row:
\`\`\`json
[
  {
    "version": "v0.9",
    "updateComponents": {
      "surfaceId": "${PLACEHOLDER_SURFACE_ID}",
      "components": [
        { "id": "root", "component": "Column", "children": ["metrics", "tools"] },
        { "id": "metrics", "component": "Row", "children": ["stat-latency", "stat-tokens"], "align": "stretch" },
        { "id": "stat-latency", "component": "StatCard", "value": "1.20s", "label": "Latency", "icon": "clock", "tone": "warning" },
        { "id": "stat-tokens", "component": "StatCard", "value": "3,120", "label": "Total Tokens", "icon": "hash", "tone": "info" },
        {
          "id": "tools",
          "component": "DataTable",
          "title": "Tool Performance",
          "icon": "wrench",
          "columns": [
            { "label": "Tool", "align": "left" },
            { "label": "Calls", "align": "center" },
            { "label": "Latency (AVG)", "align": "center" }
          ],
          "rows": [
            { "color": "#077A9D", "cells": ["run_sql_query", "4", "38.57ms"] },
            { "color": "#00A972", "cells": ["validate_schema", "4", "46.50ms"] }
          ],
          "emptyMessage": "No tool calls in this trace."
        }
      ]
    }
  }
]
\`\`\``;

const BINDING_EXAMPLE = `Example — extract span inputs/outputs from nodeMap into the data model, then bind them:
\`\`\`json
[
  {
    "version": "v0.9",
    "updateDataModel": {
      "surfaceId": "${PLACEHOLDER_SURFACE_ID}",
      "value": {
        "calls": [
          { "name": "generate_content", "input": "{\\"prompt\\":\\"Summarize\\"}", "output": "{\\"text\\":\\"...\\"}" }
        ]
      }
    }
  },
  {
    "version": "v0.9",
    "updateComponents": {
      "surfaceId": "${PLACEHOLDER_SURFACE_ID}",
      "components": [
        {
          "id": "root",
          "component": "DataTable",
          "title": "generate_content I/O",
          "columns": [ { "label": "Call" }, { "label": "Input" }, { "label": "Output" } ],
          "rows": [
            { "cells": [ { "path": "/calls/0/name" }, { "path": "/calls/0/input" }, { "path": "/calls/0/output" } ] }
          ]
        }
      ]
    }
  }
]
\`\`\`
Note how each object input/output was JSON-stringified before being placed in the data model, so it renders as text in the table cells.`;

// Caps a potentially large array for the prompt, appending a note when trimmed
// so the model knows more data exists than what it can inline.
const cap = <T>(items: T[], max: number): { items: T[]; truncated: number } => {
  if (items.length <= max) {
    return { items, truncated: 0 };
  }
  return { items: items.slice(0, max), truncated: items.length - max };
};

// Keeps small inputs/outputs structured, but truncates large payloads to a
// string so a single span can't blow up the prompt.
const truncateValue = (value: unknown, max = 500): unknown => {
  if (value === null || value === undefined) {
    return value;
  }
  const str = JSON.stringify(value);
  if (str === undefined) {
    return value;
  }
  return str.length <= max ? value : `${str.slice(0, max)}… (truncated)`;
};

export const buildAgentMessages = ({
  instruction,
  data,
}: {
  instruction: string;
  data: AgentTraceData;
}): AgentChatMessage[] => {
  const timeline = cap(data.timelineRows, 60);
  const tree = cap(data.treeNodes, 40);

  // Serialize the nodeMap for the prompt: cap the number of spans and truncate
  // each span's inputs/outputs so a large trace can't blow up the context.
  const nodeMapEntries = Object.entries(data.nodeMap ?? {});
  const cappedEntries = nodeMapEntries.slice(0, 40);
  const nodeMapJson = Object.fromEntries(
    cappedEntries.map(([id, node]) => [
      id,
      { ...node, inputs: truncateValue(node.inputs), outputs: truncateValue(node.outputs) },
    ]),
  );
  const nodeMapTruncated = Math.max(nodeMapEntries.length - cappedEntries.length, 0);

  const systemContent = [
    'You are a UI generation assistant for the MLflow trace explorer. You turn a user request into an A2UI dashboard that visualizes data from a single model trace.',
    CATALOG_REFERENCE,
    OUTPUT_RULES,
    EXAMPLE,
    BINDING_EXAMPLE,
  ].join('\n\n');

  const dataSnapshot = {
    metrics: data.metrics,
    toolRows: data.toolRows,
    timelineRows: timeline.items,
    timelineRowsTruncated: timeline.truncated,
    treeNodes: tree.items,
    treeNodesTruncated: tree.truncated,
    // Raw per-span source (including inputs/outputs), keyed by span id.
    nodeMap: nodeMapJson,
    nodeMapTruncated,
    // The trace's real assessments (LLM-judge / human feedback). This is the
    // ONLY evaluation/judge data available.
    assessments: data.assessments ?? [],
  };

  const userContent = [
    `User request: ${instruction}`,
    '',
    'Trace data is below. These precomputed arrays cover the WHOLE trace and are ready to INLINE directly (do not bind them):',
    '- `treeNodes` → TreeView "nodes" (the COMPLETE span tree)',
    '- `timelineRows` → TimelineChart "rows" (every span)',
    '- `toolRows` → DataTable rows (cells = [tool, calls, success, avg latency]) — already only the TOOL spans',
    '- `metrics` → StatCard values',
    'IMPORTANT — filtering to a subset: if the request asks for only a SUBSET of spans (e.g. "only tool calls", "only retrievers", "only errors"), do NOT inline the full array. Every tree node and every `nodeMap` entry has a "type" field (the span type: "TOOL", "LLM", "RETRIEVER", "CHAIN", "PARSER", "AGENT", etc.). Each tree node also has `attributes.type`. Select ONLY the matching nodes (e.g. type === "TOOL" for tool calls), and emit those. When you filter a TreeView, reuse each matching node\'s existing "id", "label", and "icon" from `treeNodes`, and set its "children" to [] (a flat list of the matching spans) so the icons stay correct. For tool calls specifically, you may instead use `toolRows` which already contains only tools.',
    '`assessments` is the trace\'s REAL evaluation data (LLM-judge / human feedback): each has a `name`, a `value` (e.g. "yes"/"no"/a score/boolean), an optional `rationale`, a `source` (e.g. LLM_JUDGE), and an optional `error`. For any request about judge results, evaluations, scores, or feedback, use ONLY `assessments` — there is no other scoring data. If `assessments` is empty, say it is unavailable.',
    '`nodeMap` (keyed by span id) is the raw per-span source including each span\'s `type` and `inputs`/`outputs`; use it for anything the precomputed arrays do not cover. Only individual scalar values may be bound via an `updateDataModel` message and `{ "path": "/..." }`; all arrays must be inlined.',
    '```json',
    JSON.stringify(dataSnapshot, null, 2),
    '```',
    '',
    'Generate the A2UI message stream that best satisfies the request using ONLY the data above. Do not invent any values that are not present in this data. This is a single trace; if the requested information is not available, render a short "unavailable" message instead of fabricating numbers.',
  ].join('\n');

  return [
    { role: 'system', content: systemContent },
    { role: 'user', content: userContent },
  ];
};
