/**
 * Module-level store for the Custom View authoring context (the current trace's
 * data snapshot + the active view's template). The Custom View host lives deep
 * inside the trace drawer (web-shared), while the assistant's page context is
 * assembled at a higher level; rather than thread per-trace state up through
 * every layer, the host registers the current authoring context here and the
 * assistant's context plugin reads it when building the agent prompt.
 * Last-writer-wins (a single Custom View tab is open at a time).
 */
import type { A2uiMessage } from '@a2ui/web_core/v0_9';

import type { CustomViewApplyTarget } from '../customViewDefinition';

export type CustomViewAuthoringContext = {
  currentTemplate?: A2uiMessage[];
  traceSample: Record<string, unknown>;
  // The view `currentTemplate` was taken from. Published alongside the template
  // (never derived separately) so the two can never describe different views.
  applyTarget?: CustomViewApplyTarget;
};

let currentContext: CustomViewAuthoringContext | null = null;

// The view whose template was handed to the agent on the most recent turn. See
// `latchDispatchedCustomViewApplyTarget` for why this, and not the live
// selection, decides where a spec lands.
let dispatchedApplyTarget: CustomViewApplyTarget | undefined;

export const registerCustomViewAuthoringContext = (context: CustomViewAuthoringContext): (() => void) => {
  currentContext = context;
  return () => {
    if (currentContext === context) {
      currentContext = null;
    }
  };
};

export const getCustomViewAuthoringContext = (): CustomViewAuthoringContext | null => currentContext;

/**
 * Records which view the agent is authoring against for the turn whose prompt is
 * being assembled right now. Call this wherever the authoring context is read
 * into a prompt, passing that same context's `applyTarget` (including
 * `undefined`, which clears a previous turn's latch).
 *
 * The agent edits the template it was given, so the view that template came from
 * is the view its `render_custom_view` reply belongs to — even though the reply
 * only arrives seconds later, by which time the user may have selected a
 * different view. Latching at prompt-assembly time is what makes a mid-flight
 * view switch apply the edit in the background to the right view instead of
 * overwriting whatever became active. Each turn overwrites the previous latch,
 * so it is never consumed/cleared on apply: a turn that calls the tool more than
 * once keeps the same target throughout.
 *
 * The latch's lifetime is tied to the TURN, not the host's mount. A trace
 * modal closed and reopened mid-turn is a tolerated flow (see
 * `waitForCustomViewSpecApplier`), so clearing on unregister would leave that
 * spec having lost its target, falling through to the live selection — the
 * exact overwrite-the-wrong-view failure this latch exists to prevent.
 * Passing `undefined` is for a context with no active view (an empty-state
 * build), not for the absence of a host.
 */
export const latchDispatchedCustomViewApplyTarget = (target: CustomViewApplyTarget | undefined): void => {
  dispatchedApplyTarget = target;
};

export const getDispatchedCustomViewApplyTarget = (): CustomViewApplyTarget | undefined => dispatchedApplyTarget;
