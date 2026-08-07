import type { A2uiMessage } from '@a2ui/web_core/v0_9';

// One experiment tag per saved view: `mlflow.customView.view.v1.<id>`. The
// `mlflow.` prefix keeps these out of the user-facing tag editor (see
// isUserFacingTag). Raw JSON is stored when it fits the safe limit below, and
// larger definitions are compressed before persistence. Saving fails if the
// compressed value still exceeds that limit.
export const CUSTOM_VIEW_PREFIX = 'mlflow.customView.view.v1.';
export const viewTagKey = (id: string): string => `${CUSTOM_VIEW_PREFIX}${id}`;

// The OSS backend caps an experiment tag value at MAX_EXPERIMENT_TAG_VAL_LENGTH
// (5,000 CHARACTERS, mlflow/utils/validation.py) and rejects anything longer
// outright — `_validate_experiment_tag` passes no `truncate`, so an oversized
// view fails the save rather than being silently cut. Budgeting the same number
// in UTF-8 BYTES is deliberately conservative: a string's byte length is never
// below its character count, so nothing that passes here can exceed the
// server's limit.
export const CUSTOM_VIEW_TAG_VALUE_SAFE_MAX_BYTES = 5000;
const UTF8_ENCODER = new TextEncoder();
export const getUtf8ByteLength = (value: string): number => UTF8_ENCODER.encode(value).byteLength;

// A single saved custom view. Two distinct titles:
//
// - `name` is the USER-provided name (from the "Create trace view" naming
//   modal). It's shown in the view switcher dropdown + the selected-view
//   button and is never overwritten by the assistant.
// - `label` is the assistant-generated surface title (from the assistant
//   `{title}`) shown as the panel header inside the rendered view.
// - `template` is the trace-agnostic BOUND TEMPLATE: a layout whose
//   data-bearing props are binding markers ($source / $spanRef / spanField),
//   not literal trace data. The host re-binds it to whatever trace is open via
//   resolveTemplate (no LLM call), so one stored template renders correctly on
//   every trace. Legacy data-baked templates (no markers) pass through the
//   binder unchanged.
// - `instruction` is the latest natural-language request (the human intent),
//   re-sent to the assistant as context when the user edits the view.
// - `createdAtMs` orders the switcher. No view is selected on load (the host
//   shows a placeholder until the user picks one), so this does not pick a
//   default-on-load view.
export type CustomView = {
  id: string;
  name: string;
  label: string;
  instruction: string;
  template: A2uiMessage[];
  createdAtMs: number;
  // Set at load time ONLY for a Case-1 failure: a stored tag whose bytes couldn't
  // be read into a valid shape (unparseable deflate/JSON, or a structural
  // mismatch), flagged by the loader (`deserializeView`). A structurally-valid
  // view whose TEMPLATE fails validation (Case 2) is NOT flagged here — that is
  // derived lazily when the view becomes active (see `CustomViewDefinitionContext`
  // + the render-time gate) so load doesn't validate every saved view. Either way
  // such a view stays in the switcher (keyed by its tag id), renders a "couldn't
  // be read" placeholder, and has rename/save blocked so the invalid definition
  // isn't re-persisted.
  unreadable?: boolean;
};

// The identity of the view an in-flight assistant request will apply to,
// captured up front so a spec that lands later still reaches the view the
// request was made against rather than whatever happens to be selected then.
// Carries the metadata `onSpec` must preserve (the agent authors only the
// template and the `label`), so the target stays usable even if the user
// navigates away while the agent runs.
export type CustomViewApplyTarget = Pick<CustomView, 'id' | 'name' | 'instruction' | 'createdAtMs'> &
  Partial<Pick<CustomView, 'label'>>;

export const toCustomViewApplyTarget = (view: CustomView): CustomViewApplyTarget => ({
  id: view.id,
  name: view.name,
  label: view.label,
  instruction: view.instruction,
  createdAtMs: view.createdAtMs,
});

// Narrow an untrusted parsed object into a CustomView. This is purely a SHAPE
// narrower: it does NOT validate the template (that walk is deferred to the
// selection/render path so tab load doesn't validate every saved view). The
// template is kept verbatim — the render-time gate re-validates it per trace and
// shows a placeholder rather than rendering, and preserving it also gives the
// assistant a repair seed (`useCustomViewAssistantBridge` publishes the active
// view's template as edit context). Returns undefined only for a structural
// mismatch (non-object, missing string `id`, non-array `template`); the loader
// (`deserializeView`) then keeps even those as an `unreadable` placeholder keyed
// by the tag id, so nothing but the soft-delete tombstone drops from the list.
export const parseCustomView = (value: unknown): CustomView | undefined => {
  if (!value || typeof value !== 'object') {
    return undefined;
  }
  const candidate = value as Partial<CustomView>;
  if (typeof candidate.id !== 'string' || !Array.isArray(candidate.template)) {
    return undefined;
  }
  // Guard once and reuse for the label fallback: a non-string `name` (untrusted
  // tag JSON) must not leak through `label` and reach the renderer as a
  // non-primitive React child.
  const name = typeof candidate.name === 'string' ? candidate.name : '';
  return {
    id: candidate.id,
    name,
    label: typeof candidate.label === 'string' ? candidate.label : name,
    instruction: typeof candidate.instruction === 'string' ? candidate.instruction : '',
    template: candidate.template,
    createdAtMs: typeof candidate.createdAtMs === 'number' ? candidate.createdAtMs : 0,
  };
};

export const serializeCustomView = (view: CustomView): string => JSON.stringify(view);
