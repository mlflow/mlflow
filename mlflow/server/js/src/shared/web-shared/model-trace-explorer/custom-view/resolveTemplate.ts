// The host-side binder. Given a trace-agnostic template (authored once by the
// LLM, with `$source` / `$spanRef` markers in place of data) and the CURRENT
// trace's data, it returns a concrete A2UI message stream with every marker
// resolved - NO LLM call. This is what lets a saved custom view re-render for
// every cycled trace by swapping data while preserving the authored layout.
//
// The output still uses the template's placeholder surfaceId; callers pass it
// through `validateAndPrepareMessages` to inject the host-owned createSurface,
// rewrite the surfaceId, and strict-validate the resolved components.

import type { A2uiMessage } from '@a2ui/web_core/v0_9';

import type { ModelTraceSpanNode } from '../ModelTrace.types';
import { type CustomViewData, buildAssessmentCardComponents } from './customViewBuilders';
import {
  isRecord,
  isScalarSource,
  isSourceMarker,
  isSpanFieldSource,
  isSpanRefMarker,
  isStructuralSource,
  isValidSpanRefSelector,
  resolveScalarSource,
  resolveSpanFieldSource,
  resolveSpanRef,
  type SourceMarker,
  unwrapSpanRefSelector,
} from './customViewSources';

const SPAN_GUARD_KEY = 'renderIfSpan';
const SPAN_TARGETED_FEEDBACK_COMPONENTS = new Set(['FeedbackThumbsUpDownButtons', 'RadioGroup', 'FeedbackInputText']);

// The component id the renderer walks the tree from, and which
// `validateAndPrepareMessages` requires the resolved stream to contain.
const ROOT_COMPONENT_ID = 'root';

export type ResolveContext = {
  viewData: CustomViewData;
  nodeMap: Record<string, ModelTraceSpanNode>;
};

// Resolves scalar / spanField / spanRef markers anywhere in a (possibly nested)
// value.
const resolveValueDeep = (value: unknown, ctx: ResolveContext): unknown => {
  if (isSourceMarker(value)) {
    const name = value['$source'];
    if (isScalarSource(name)) {
      return resolveScalarSource(name, ctx.viewData) ?? '';
    }
    if (isSpanFieldSource(name)) {
      return resolveSpanFieldSource(value, ctx.nodeMap);
    }
    // A structural marker used outside a `children` slot has no meaning; drop it.
    return '';
  }
  if (isSpanRefMarker(value)) {
    // Nested inside a value, an unresolved ref degrades to "" like every other
    // source here — NOT to the dropped prop `resolveComponent` produces for a
    // top-level marker. There the marker is a control TARGET, so absent must
    // mean absent; here it is display data, and dropping it is not even
    // expressible (an array entry has no key), let alone safe: required props
    // like KeyValueViewer's `value` would fail the strict resolved-output
    // validation and error the whole view instead of rendering empty.
    return resolveSpanRef(value.$spanRef, ctx.nodeMap) ?? '';
  }
  if (Array.isArray(value)) {
    return value.map((entry) => resolveValueDeep(entry, ctx));
  }
  if (isRecord(value)) {
    return Object.fromEntries(Object.entries(value).map(([key, entry]) => [key, resolveValueDeep(entry, ctx)]));
  }
  return value;
};

// Materializes a structural source marker into child component ids + the
// components to append after the owner.
const materializeStructural = (
  marker: SourceMarker,
  ownerId: string,
  ctx: ResolveContext,
): { childIds: string[]; components: Record<string, unknown>[] } => {
  if (marker.$source === 'assessments') {
    return buildAssessmentCardComponents(ctx.viewData.assessmentItems, { idPrefix: `${ownerId}__a` });
  }
  return { childIds: [], components: [] };
};

const resolveComponent = (
  component: Record<string, unknown>,
  ctx: ResolveContext,
): { component: Record<string, unknown>; generated: Record<string, unknown>[] } => {
  const ownerId = typeof component['id'] === 'string' ? component['id'] : String(component['id'] ?? '');
  const resolved: Record<string, unknown> = {};
  const generated: Record<string, unknown>[] = [];

  for (const [key, value] of Object.entries(component)) {
    if (key === 'id' || key === 'component') {
      resolved[key] = value;
      continue;
    }
    // Host-only directive consumed during pruning; never emit it (the strict
    // resolved-output validator would reject it as an unknown prop).
    if (key === SPAN_GUARD_KEY) {
      continue;
    }
    if (isSourceMarker(value) && isStructuralSource(value['$source'])) {
      const { childIds, components } = materializeStructural(value, ownerId, ctx);
      resolved[key] = childIds;
      generated.push(...components);
      continue;
    }
    if (isSpanRefMarker(value)) {
      const spanId = resolveSpanRef(value.$spanRef, ctx.nodeMap);
      // Missing feedback targets are pruned before resolution. For any other
      // top-level marker, omit an unresolved prop instead of emitting an invalid
      // span id.
      if (spanId) {
        resolved[key] = spanId;
      }
      continue;
    }
    resolved[key] = resolveValueDeep(value, ctx);
  }

  return { component: resolved, generated };
};

// Computes the set of component ids to prune: every component carrying a
// `renderIfSpan` guard that resolves to NO span in the current trace, plus every
// span-targeted feedback control whose `spanId` marker cannot resolve, and all
// of their descendants (walked via `child`/`children`). An invalid/unrecognized
// marker is ignored here because template validation rejects it before this
// resolver runs.
//
// We then cascade upward across the singular `child` link: a component whose
// sole `child` was pruned is left empty, and for the strict-validated Card
// (whose `child` is required) that would emit an invalid childless component
// instead of collapsing the card. So a component whose `child` target is pruned
// is pruned too, iterated to a fixpoint so a collapsed card that is itself
// another card's child also drops. `children` arrays are left alone — removing
// individual entries there always yields a valid container.
const computePrunedIds = (components: unknown[], nodeMap: Record<string, ModelTraceSpanNode>): Set<string> => {
  const byId = new Map<string, Record<string, unknown>>();
  for (const component of components) {
    if (isRecord(component) && typeof component['id'] === 'string') {
      byId.set(component['id'], component);
    }
  }
  const pruned = new Set<string>();
  const markSubtree = (id: string) => {
    if (pruned.has(id)) {
      return;
    }
    pruned.add(id);
    const component = byId.get(id);
    if (!component) {
      return;
    }
    // Follow BOTH child-reference shapes: a Card nests its single child via
    // "child" (string); Row/Column/AssessmentBoard/etc. via "children" (string ids).
    if (typeof component['child'] === 'string') {
      markSubtree(component['child']);
    }
    if (Array.isArray(component['children'])) {
      for (const child of component['children']) {
        if (typeof child === 'string') {
          markSubtree(child);
        }
      }
    }
  };
  for (const component of components) {
    if (!isRecord(component) || typeof component['id'] !== 'string') {
      continue;
    }
    const guardedSelector = unwrapSpanRefSelector(component[SPAN_GUARD_KEY]);
    const feedbackSpanMarker = component['spanId'];
    const feedbackSelector = isSpanRefMarker(feedbackSpanMarker) ? feedbackSpanMarker.$spanRef : undefined;
    const hasMissingGuard =
      SPAN_GUARD_KEY in component &&
      isValidSpanRefSelector(guardedSelector) &&
      !resolveSpanRef(guardedSelector, nodeMap);
    const hasMissingFeedbackTarget =
      SPAN_TARGETED_FEEDBACK_COMPONENTS.has(String(component['component'])) &&
      isValidSpanRefSelector(feedbackSelector) &&
      !resolveSpanRef(feedbackSelector, nodeMap);
    if (hasMissingGuard || hasMissingFeedbackTarget) {
      markSubtree(component['id']);
    }
  }
  // Cascade the prune up the singular `child` link.
  let changed = true;
  while (changed) {
    changed = false;
    for (const [id, component] of byId) {
      if (pruned.has(id)) {
        continue;
      }
      if (typeof component['child'] === 'string' && pruned.has(component['child'])) {
        markSubtree(id);
        changed = true;
      }
    }
  }
  return pruned;
};

// Walks an `updateComponents` payload's components, resolving every component's
// markers and appending any structurally-materialized children (so parents still
// precede children in the list). Components pruned by a `renderIfSpan` guard (and
// their descendants) are dropped, and their ids are removed from any parent's
// `children` so the layout closes up cleanly. A prune that takes out the root
// collapses the view to an empty root rather than removing it. Returns the
// resolved component list (the only part of the payload this transforms).
const resolveComponents = (payload: Record<string, unknown>, ctx: ResolveContext): Record<string, unknown>[] => {
  const components = Array.isArray(payload['components']) ? payload['components'] : [];
  const pruned = computePrunedIds(components, ctx.nodeMap);
  const resolved: Record<string, unknown>[] = [];
  const generated: Record<string, unknown>[] = [];
  for (const component of components) {
    if (!isRecord(component)) {
      continue;
    }
    const id = typeof component['id'] === 'string' ? component['id'] : String(component['id'] ?? '');
    if (pruned.has(id)) {
      continue;
    }
    const result = resolveComponent(component, ctx);
    resolved.push(result.component);
    generated.push(...result.generated);
  }
  if (pruned.size > 0) {
    for (const component of resolved) {
      if (Array.isArray(component['children'])) {
        component['children'] = component['children'].filter(
          (child: unknown) => !(typeof child === 'string' && pruned.has(child)),
        );
      }
      if (typeof component['child'] === 'string' && pruned.has(component['child'])) {
        delete component['child'];
      }
    }
    // The root can be pruned like any other component — by a guard on itself, or
    // by the `child` cascade reaching it — but a stream with no root is rejected
    // WHOLESALE by `validateAndPrepareMessages`, so the view would surface a
    // render error instead of hiding the guarded content. Substitute an empty
    // root so it collapses to a blank view. The pruned root's own definition
    // can't be reused: a Card that lost its required `child` fails the strict
    // resolved-output validation, so emit a bare Column.
    if (
      pruned.has(ROOT_COMPONENT_ID) &&
      components.some((component) => isRecord(component) && component['id'] === ROOT_COMPONENT_ID)
    ) {
      resolved.unshift({ id: ROOT_COMPONENT_ID, component: 'Column', children: [] });
    }
  }
  return [...resolved, ...generated];
};

/**
 * Resolves a stored bound template against the current trace. The result is a
 * raw A2UI message stream (markers replaced with this trace's data/ids) ready to
 * hand to `validateAndPrepareMessages`. A template with no markers (e.g. a legacy
 * data-baked view) passes through unchanged.
 */
export const resolveTemplate = (template: A2uiMessage[], ctx: ResolveContext): A2uiMessage[] =>
  template.map((rawMessage) => {
    // `in` narrows the A2uiMessage union to UpdateComponentsMessage, so the
    // reconstructed object is typed without an assertion.
    if ('updateComponents' in rawMessage && isRecord(rawMessage.updateComponents)) {
      return {
        ...rawMessage,
        updateComponents: {
          ...rawMessage.updateComponents,
          components: resolveComponents(rawMessage.updateComponents, ctx),
        },
      };
    }
    return rawMessage;
  });
