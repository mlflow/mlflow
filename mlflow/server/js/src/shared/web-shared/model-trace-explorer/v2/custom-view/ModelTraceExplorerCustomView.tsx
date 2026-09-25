import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from 'react';

import { uniqBy } from 'lodash';

import {
  Alert,
  Button,
  DropdownMenu,
  Empty,
  GenericSkeleton,
  InfoIcon,
  Input,
  Modal,
  OverflowIcon,
  PencilIcon,
  Tooltip,
  TrashIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { Catalog, MessageProcessor, type A2uiClientAction, type A2uiMessage } from '@a2ui/web_core/v0_9';
import { BASIC_FUNCTIONS } from '@a2ui/web_core/v0_9/basic_catalog';
import { A2uiSurface, Column, Row } from '@a2ui/react/v0_9';

import type { Feedback, ModelTrace } from '../ModelTrace.types';
import { ModelSpanType } from '../ModelTrace.types';
import { isV3ModelTraceInfo } from '../ModelTraceExplorer.utils';
import { useModelTraceExplorerContext } from '../ModelTraceExplorerContext';
import { ModelTraceExplorerAssistantButton } from '../ModelTraceExplorerAssistantButton';
import type { CreateAssessmentPayload } from '../../api';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';
import { getUser } from '../../../global-settings/getUser';
import { shouldUseTracesV4API } from '../../FeatureUtils';
import { useCreateAssessment } from '../../hooks/useCreateAssessment';
import { useTraceCachedActions } from '../../hooks/useTraceCachedActions';
import { AssessmentBoard } from '../../custom-view/catalog-primitives/AssessmentBoard';
import { AssessmentCard } from '../../custom-view/catalog-primitives/AssessmentCard';
import { Card } from '../../custom-view/catalog-primitives/Card';
import { FeedbackThumbsUpDownButtons } from './catalog-primitives/FeedbackThumbsUpDownButtons';
import { FEEDBACK_STAGED } from '../../custom-view/catalog-primitives/feedbackActions';
import { FeedbackInputText } from '../../custom-view/catalog-primitives/FeedbackInputText';
import { FeedbackSubmit } from '../../custom-view/catalog-primitives/FeedbackSubmit';
import { RadioGroup } from '../../custom-view/catalog-primitives/RadioGroup';
import { FeedbackStatusProvider } from '../../custom-view/FeedbackStatusContext';
import { Icon } from '../../custom-view/catalog-primitives/Icon';
import { KeyValueViewer } from '../../custom-view/catalog-primitives/KeyValueViewer';
import { Markdown } from '../../custom-view/catalog-primitives/Markdown';
import { StatCard } from '../../custom-view/catalog-primitives/StatCard';
import { Text } from '../../custom-view/catalog-primitives/Text';
import type { AgentNode } from '../../custom-view/agent/buildAgentPrompt';
import { validateAndPrepareMessages, validateTemplate } from '../../custom-view/agent/validateA2uiMessages';
import { resolveTemplate } from '../../custom-view/resolveTemplate';
import { useCustomViewAssistantBridge } from '../../custom-view/assistant/useCustomViewAssistantBridge';
import { getDispatchedCustomViewApplyTarget } from '../../custom-view/assistant/customViewAuthoringContext';
import type { RenderCustomViewSpec } from '../../custom-view/assistant/customViewSpecApplier';
import {
  CUSTOM_VIEW_CATALOG_ID,
  type CustomViewData,
  collectTraceAssessments,
  getAssessmentBoardItems,
  getMetricsFromTraceInfo,
  mapToAgentAssessments,
} from '../../custom-view/customViewBuilders';
import { type CustomView, type CustomViewApplyTarget } from '../../custom-view/customViewDefinition';
import { useCustomViewDefinition } from '../../custom-view/CustomViewDefinitionContext';

// Deterministic surface id per view so React/A2UI reuse the same surface across
// trace cycling (we rebuild the surface contents, not its identity).
const surfaceIdForView = (view: CustomView): string => `cv-${view.id}`;

// Buffer key for a staged feedback entry: formId + name + spanId, NUL-separated
// so the same dimension in different forms (or on different spans) stays
// distinct. A radio and its rationale input share a key only when they share
// BOTH formId and name. Single source of truth so staging, reset-versioning, and
// the value getters all key identically.
const feedbackBufferKey = (entry: { name: string; spanId?: string; formId?: string }): string =>
  `${entry.formId ?? ''}\u0000${entry.name}\u0000${entry.spanId ?? ''}`;

let viewIdCounter = 0;
const nextViewId = (): string => `${Date.now().toString(36)}-${(viewIdCounter++).toString(36)}`;

// A staged entry belongs to a submit when they share the same form. `formId`
// is the ownership key (additive to `spanId`, which stays the assessment's span
// target): a submit flushes ONLY its own form's entries, never another form's,
// even when two forms rate the same span. An omitted `formId` on both sides is
// the implicit default form, so a bare single-form view needs no formId at all.
const doesFeedbackEntryMatchForm = (entry: { formId?: string }, formId: string | undefined): boolean =>
  entry.formId === formId;

const placeholderMessages = (surfaceId: string, text: string): A2uiMessage[] => [
  { version: 'v0.9', createSurface: { surfaceId, catalogId: CUSTOM_VIEW_CATALOG_ID, sendDataModel: true } },
  { version: 'v0.9', updateComponents: { surfaceId, components: [{ id: 'root', component: 'Text', text }] } },
];

// Host-rendered loading state shown in the empty state while the agent streams
// the FIRST authoring reply (before any view exists). Per-trace cycling re-binds
// host-side with no LLM call, so there is no per-trace loading state.
const CustomViewGeneratingSkeleton = () => {
  const { theme } = useDesignSystemTheme();
  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md, width: '100%', maxWidth: 640 }}>
      <Typography.Text color="secondary">
        <FormattedMessage
          defaultMessage="Building this view…"
          description="Loading message shown while the assistant builds the custom trace view"
        />
      </Typography.Text>
      <div css={{ display: 'flex', gap: theme.spacing.md }}>
        <GenericSkeleton css={{ height: 80, flex: 1 }} />
        <GenericSkeleton css={{ height: 80, flex: 1 }} />
        <GenericSkeleton css={{ height: 80, flex: 1 }} />
      </div>
      <GenericSkeleton css={{ height: 160, width: '100%' }} />
      <GenericSkeleton css={{ height: 120, width: '100%' }} />
    </div>
  );
};

/**
 * Custom View tab: renders the active view's A2UI surface for the open trace and
 * routes all authoring through the assistant. The empty-state box builds the first view;
 * "Edit with Assistant" reopens the assistant to change it. The agent authors a
 * trace-agnostic BOUND TEMPLATE once (via render_custom_view -> onSpec); cycling
 * traces re-binds it host-side with no further LLM call.
 */
export const ModelTraceExplorerCustomView = ({
  modelTraceInfo,
}: {
  modelTraceInfo: ModelTrace['info'];
}): JSX.Element => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const { nodeMap } = useModelTraceExplorerViewState();
  const { setTraceExplorerDisplayMode } = useModelTraceExplorerContext();

  const cv = useCustomViewDefinition();

  const activeView = cv.activeView;

  // The catalog maps component type names to their React implementations: the
  // layout/content primitives plus the feedback controls a bound template can
  // reference
  const catalog = useMemo(
    () =>
      new Catalog(
        CUSTOM_VIEW_CATALOG_ID,
        [
          Text,
          Row,
          Column,
          Card,
          Icon,
          StatCard,
          Markdown,
          AssessmentBoard,
          AssessmentCard,
          KeyValueViewer,
          FeedbackThumbsUpDownButtons,
          RadioGroup,
          FeedbackInputText,
          FeedbackSubmit,
        ],
        BASIC_FUNCTIONS,
      ),
    [],
  );

  const traceId = useMemo(
    () => (isV3ModelTraceInfo(modelTraceInfo) ? modelTraceInfo.trace_id : (modelTraceInfo.request_id ?? '')),
    [modelTraceInfo],
  );
  const { createAssessmentMutation } = useCreateAssessment({ traceId });

  // The processor's action handler is created once, so we route through a ref
  // that always points at the latest mutation / traceId / nodeMap.
  const actionHandlerRef = useRef<(action: A2uiClientAction) => void>(() => {});

  // Staged-but-unsubmitted feedback per surface, keyed by `formId` + `name` +
  // `spanId` so the same dimension rated in two forms, or on two spans (e.g.
  // "Accuracy" on two tool cards), does not collide. `formId` is which form
  // owns the entry — a FeedbackSubmit flushes only its own form. The RadioGroup
  // / FeedbackInputText primitives merge their value/rationale here (no POST);
  // FeedbackSubmit flushes the form's entries into assessments.
  const pendingFeedbackRef = useRef<
    Map<string, Map<string, { name: string; value?: string; rationale?: string; spanId?: string; formId?: string }>>
  >(new Map());

  // Successful entries increment their own reset version (keyed identically to
  // pendingFeedbackRef) so the matching radio / text controls clear their visible
  // value after persistence, while failed or newly edited entries stay visible
  // for retry. Mirrors the v1 wrapper's feedbackResetVersionsRef.
  const feedbackResetVersionsRef = useRef<Map<string, Map<string, number>>>(new Map());

  // Ticks whenever the pending-feedback buffer changes (stage, submit, clear).
  // Lets React re-render the FeedbackStatusProvider so `hasStagedFeedback` can
  // reflect the current buffer state (the ref alone is not reactive).
  const [pendingFeedbackVersion, bumpPendingFeedbackVersion] = useReducer((tick: number) => tick + 1, 0);

  // A single long-lived processor holds the state for the active view's surface.
  const [processor] = useState(
    () => new MessageProcessor([catalog], (action: A2uiClientAction) => actionHandlerRef.current(action)),
  );

  // Persists one assessment and resolves/rejects on the real request outcome, so
  // the awaiting FeedbackSubmit flush can keep failed dimensions staged and count
  // the successes. `mutate` accepts per-call callbacks (react-query v4); the
  // hook's own onError still fires its global error notification underneath.
  // FeedbackSubmit awaits these ONE AT A TIME (see submitStagedFeedback), so the
  // single shared MutationObserver never has two overlapping calls whose per-call
  // callbacks would clobber each other. Thumbs do NOT use this path — each
  // FeedbackThumbsUpDownButtons owns its own useCreateAssessment instance.
  const createAssessmentAsync = (payload: CreateAssessmentPayload): Promise<void> =>
    new Promise<void>((resolve, reject) => {
      createAssessmentMutation(payload, {
        onSuccess: () => resolve(),
        onError: (error) => reject(error instanceof Error ? error : new Error(String(error))),
      });
    });

  // Merge a staged-feedback change into the surface's pending buffer (no POST).
  // `action.context` comes from an untrusted A2UI client action, so every field
  // is read defensively rather than asserted via a cast.
  const handleStageFeedback = (action: A2uiClientAction) => {
    const context = action.context ?? {};
    const name = typeof context['name'] === 'string' && context['name'] ? context['name'] : undefined;
    if (!name) {
      return;
    }
    let surfaceBuffer = pendingFeedbackRef.current.get(action.surfaceId);
    if (!surfaceBuffer) {
      surfaceBuffer = new Map();
      pendingFeedbackRef.current.set(action.surfaceId, surfaceBuffer);
    }
    const spanId = typeof context['spanId'] === 'string' && context['spanId'] ? context['spanId'] : undefined;
    const formId = typeof context['formId'] === 'string' && context['formId'] ? context['formId'] : undefined;
    // Key by formId + name + spanId so the same dimension in different forms (or
    // on different spans) stays distinct. A radio and its rationale input share a
    // key only when they share BOTH formId and name — that pairing is intentional.
    const bufferKey = `${formId ?? ''}\u0000${name}\u0000${spanId ?? ''}`;
    const previous = surfaceBuffer.get(bufferKey) ?? { name };
    surfaceBuffer.set(bufferKey, {
      ...previous,
      name,
      ...(typeof context['value'] === 'string' ? { value: context['value'] } : {}),
      ...(typeof context['rationale'] === 'string' ? { rationale: context['rationale'] } : {}),
      ...(spanId ? { spanId } : {}),
      ...(formId ? { formId } : {}),
    });
    // Buffer changed; re-render so FeedbackSubmit's disabled state re-evaluates.
    bumpPendingFeedbackVersion();
  };

  // Flush this form's staged dimensions on the ACTIVE surface into assessments
  // (one per name). Only entries sharing the button's `formId` are flushed, so a
  // form's submit never logs another form's ratings. A rationale WITHOUT
  // a value is skipped — the server
  // requires a value (or an explicit error) on every feedback assessment, so a
  // rationale-only entry has no valid shape to persist; the user must pick a
  // rating for the rationale to be logged with it. Rejects when nothing could
  // be submitted OR every attempt failed so the button can surface an error.
  const submitStagedFeedback = async (formId?: string): Promise<{ submitted: number }> => {
    const surfaceId = activeView ? surfaceIdForView(activeView) : undefined;
    const surfaceBuffer = surfaceId ? pendingFeedbackRef.current.get(surfaceId) : undefined;
    if (!surfaceId || !surfaceBuffer || surfaceBuffer.size === 0) {
      throw new Error('There is no staged feedback to submit.');
    }
    // Snapshot the submittable entries (buffer key + payload), dropping
    // rationale-only entries (see the docstring above for why) and entries not
    // owned by this button's form. Keeping the key lets us remove each entry from
    // the buffer only AFTER it succeeds, so a failed dimension stays staged and
    // the user can retry it.
    const submittable: {
      key: string;
      entry: { name: string; value?: string; rationale?: string; spanId?: string; formId?: string };
      payload: CreateAssessmentPayload;
    }[] = [];
    for (const [key, entry] of surfaceBuffer.entries()) {
      if (!doesFeedbackEntryMatchForm(entry, formId)) {
        continue;
      }
      const hasValue = typeof entry.value === 'string' && entry.value.length > 0;
      if (!hasValue) {
        continue;
      }
      const hasRationale = typeof entry.rationale === 'string' && entry.rationale.length > 0;
      const feedbackValue: { feedback: Feedback } = { feedback: { value: entry.value } };
      submittable.push({
        key,
        entry,
        payload: {
          assessment: {
            assessment_name: entry.name,
            trace_id: traceId,
            source: { source_type: 'HUMAN', source_id: getUser() ?? '' },
            ...(entry.spanId ? { span_id: entry.spanId } : {}),
            ...feedbackValue,
            ...(hasRationale ? { rationale: entry.rationale } : {}),
          },
        },
      });
    }
    if (submittable.length === 0) {
      throw new Error('There is no staged feedback with a value to submit.');
    }
    // Fire sequentially: react-query v4's MutationObserver stores the per-call
    // { onSuccess, onError } options in a SINGLE field, so concurrent mutate()
    // calls clobber each other — only the last call's callbacks fire, leaving
    // earlier promises hanging forever. Awaiting each in turn keeps the observer
    // one-call-at-a-time. On success, drop that entry from the buffer; on failure
    // leave it staged (the hook already showed its global error toast) so the
    // user can retry only the dimensions that didn't land.
    let submitted = 0;
    let failed = 0;
    for (const { key, entry, payload } of submittable) {
      try {
        await createAssessmentAsync(payload);
        submitted += 1;
        // A trace/view transition can replace the surface's buffer while this
        // request is in flight, and the user can edit the same control before it
        // finishes. Clear the entry and bump its reset version only when both the
        // buffer and the entry are still the exact versions captured for this
        // request, so a newer edit or a different buffer is never clobbered.
        if (pendingFeedbackRef.current.get(surfaceId) === surfaceBuffer && surfaceBuffer.get(key) === entry) {
          surfaceBuffer.delete(key);
          let resetVersions = feedbackResetVersionsRef.current.get(surfaceId);
          if (!resetVersions) {
            resetVersions = new Map();
            feedbackResetVersionsRef.current.set(surfaceId, resetVersions);
          }
          resetVersions.set(key, (resetVersions.get(key) ?? 0) + 1);
        }
      } catch {
        // Keep the entry staged for retry; continue with the remaining dimensions.
        failed += 1;
      }
    }
    if (pendingFeedbackRef.current.get(surfaceId) === surfaceBuffer && surfaceBuffer.size === 0) {
      pendingFeedbackRef.current.delete(surfaceId);
    }
    bumpPendingFeedbackVersion();
    // Reject if ANY dimension failed so a partial failure is not presented as a
    // fully successful submit; failed entries remain staged above for retry.
    if (failed > 0) {
      throw new Error(
        submitted > 0 ? 'Some staged feedback requests failed.' : 'Every staged feedback request failed.',
      );
    }
    return { submitted };
  };

  actionHandlerRef.current = (action: A2uiClientAction) => {
    if (action.name === FEEDBACK_STAGED) {
      handleStageFeedback(action);
    }
  };

  // Feedback submitted in this view (thumbs / form) doesn't mutate the static
  // `modelTraceInfo` / `nodeMap` props, so derive the assessments from the base
  // trace MERGED with the trace-cached-actions store (the create-assessment hook
  // logs each add/delete there). This keeps the counter metric and the
  // AssessmentBoard live after a submit — matching the assessments pane, which
  // reads the same store. Merge is V4-only (the store is only written under V4).
  const reconstructAssessments = useTraceCachedActions((state) => state.reconstructAssessments);
  const cachedActions = useTraceCachedActions((state) => state.assessmentActions[traceId]);
  const agentAssessments = useMemo(() => {
    const base = collectTraceAssessments(modelTraceInfo, nodeMap);
    if (!shouldUseTracesV4API()) {
      return mapToAgentAssessments(base);
    }
    const merged = uniqBy(reconstructAssessments(base, cachedActions), ({ assessment_id }) => assessment_id);
    return mapToAgentAssessments(merged);
  }, [modelTraceInfo, nodeMap, reconstructAssessments, cachedActions]);
  const assessmentItems = useMemo(() => getAssessmentBoardItems(agentAssessments), [agentAssessments]);

  // The counter counts the same set the board shows (post-merge), so the two
  // never disagree and both update together on submit.
  const metrics = useMemo(
    () => getMetricsFromTraceInfo(modelTraceInfo, assessmentItems.length),
    [modelTraceInfo, assessmentItems.length],
  );

  const viewData = useMemo<CustomViewData>(() => ({ metrics, assessmentItems }), [metrics, assessmentItems]);

  // The trace's nodeMap as plain JSON (keyed by span id) for the assistant.
  const agentNodeMap = useMemo(() => {
    const nodes = Object.values(nodeMap);
    if (nodes.length === 0) {
      return {};
    }
    // Reduce instead of Math.min(...map): spreading every span's start as
    // function args throws RangeError (max call stack) on very large traces
    // (tens of thousands of spans).
    let traceStartUs = Infinity;
    for (const node of nodes) {
      if (node.start < traceStartUs) {
        traceStartUs = node.start;
      }
    }
    const json: Record<string, AgentNode> = {};
    for (const node of nodes) {
      json[String(node.key)] = {
        name: typeof node.title === 'string' ? node.title : String(node.title ?? 'unknown'),
        type: node.type ?? ModelSpanType.UNKNOWN,
        startMs: Math.max(node.start - traceStartUs, 0) / 1000,
        endMs: Math.max(node.end - traceStartUs, 0) / 1000,
        durationMs: Math.max(node.end - node.start, 0) / 1000,
        parentId: node.parentId ? String(node.parentId) : undefined,
        inputs: node.inputs,
        outputs: node.outputs,
      };
    }
    return json;
  }, [nodeMap]);

  // The full trace data handed to the assistant bridge. Memoized so its
  // reference is stable across renders (it only changes when the active trace's
  // data changes).
  const agentData = useMemo(
    () => ({ ...viewData, nodeMap: agentNodeMap, assessments: agentAssessments }),
    [viewData, agentNodeMap, agentAssessments],
  );

  // The prompt typed in the empty-state box before/while a view is being built.
  const [instruction, setInstruction] = useState('');

  const isInitialBuilding = cv.isBuilding;
  const { stopBuilding } = cv;

  // The naming modal collects the user-facing view name (distinct from the
  // agent-generated panel label).
  const [nameModalOpen, setNameModalOpen] = useState(false);
  const [nameInput, setNameInput] = useState('');
  const [renameModalOpen, setRenameModalOpen] = useState(false);
  const [renameInput, setRenameInput] = useState('');
  // The view the open rename modal targets, captured when it opens. The confirm
  // handler renames THIS id, not whatever is active at confirm time, so a
  // background selection change (e.g. an assistant apply) can't retarget the rename.
  const [renameTargetId, setRenameTargetId] = useState<string | undefined>(undefined);
  const [deleteModalOpen, setDeleteModalOpen] = useState(false);

  const managedSurfacesRef = useRef<Set<string>>(new Set());

  // The id of the view targeted by the next agent spec. Held in a ref so that
  // specs applied in the same tick all resolve to the same id. Kept in sync with
  // the active view's id.
  const draftViewIdRef = useRef<string | undefined>(cv.activeViewId);
  useEffect(() => {
    draftViewIdRef.current = cv.activeViewId;
  }, [cv.activeViewId]);

  // The target for a build launched from the empty-state box, bound at launch
  // (see handleSubmitPrompt). That build has no active view for the authoring
  // context to describe, so this ref is what reserves the new view's identity;
  // for an EXISTING view the authoring-context latch is authoritative instead
  // (see onSpec). Consumed (cleared) by the spec it launched.
  const pendingApplyTargetRef = useRef<CustomViewApplyTarget | undefined>(undefined);

  // The rebuild effect mutates the processor model AFTER render (creating new
  // surface objects). We render by reading the SurfaceModel out of the processor
  // model, so we force one render afterwards to re-read the current surface.
  // Intentionally NOT a rebuild-effect dependency, so it never re-runs the
  // rebuild (no loop).
  const [, refreshSurfaces] = useReducer((tick: number) => tick + 1, 0);

  // Parses + validates a raw agent spec into a stored, trace-agnostic BOUND
  // TEMPLATE (its `$source` / `$spanRef` markers preserved). Throws a descriptive
  // Error on failure. The template is re-bound per trace at render time by
  // `resolveTemplateForTrace` — no further LLM call. `validateTemplate` enforces
  // the feedback-form rules (every control/submit has a `formId`, every submit
  // owns a control) on this authoring path, so a bad form becomes a retryable
  // error the agent can fix.
  const prepareTemplate = (spec: RenderCustomViewSpec): A2uiMessage[] => {
    const result = validateTemplate(spec);
    if (!result.ok) {
      throw new Error(result.error);
    }
    return result.messages;
  };

  // Re-binds a stored template to the CURRENT trace: resolves every $source /
  // $spanRef marker against this trace's data, then strict-validates the resolved
  // components and stamps the host surface id. Falls back to an inline error
  // placeholder if the resolved stream fails validation. Memoized on its data
  // inputs so the rebuild effect can list it as a stable dependency (its identity
  // only changes when the trace data or locale it closes over changes).
  const resolveTemplateForTrace = useCallback(
    (template: A2uiMessage[], surfaceId: string): A2uiMessage[] => {
      // Re-run the marker-aware validation on every re-bind (not just at load /
      // authoring): the template originates from an untrusted experiment tag, so
      // this is the last gate before the renderer against a tampered stored view.
      // The feedback-form rules (formId presence + submit pairing) are enforced
      // here too, so a saved view that predates `formId` fails closed to the
      // inline placeholder rather than rendering a broken form.
      const gate = validateTemplate(template);
      if (!gate.ok) {
        return placeholderMessages(
          surfaceId,
          intl.formatMessage({
            defaultMessage:
              "This view's definition couldn't be read and can't be displayed. Edit it with the assistant to rebuild it.",
            description:
              'Placeholder shown when a custom view has an invalid or unreadable definition and cannot be rendered',
          }),
        );
      }
      const resolved = resolveTemplate(gate.messages, { viewData, nodeMap });
      const result = validateAndPrepareMessages(resolved, { surfaceId, catalogId: CUSTOM_VIEW_CATALOG_ID });
      if (!result.ok) {
        return placeholderMessages(
          surfaceId,
          intl.formatMessage(
            {
              defaultMessage: 'Could not render this view for the current trace: {error}',
              description: 'Inline error shown when a saved custom view fails to render for the open trace',
            },
            { error: result.error },
          ),
        );
      }
      return result.messages;
    },
    [viewData, nodeMap, intl],
  );

  const assistant = useCustomViewAssistantBridge({
    data: agentData,
    activeView,
    enabled: cv.canPersist,
    onSpec: (spec) => {
      if (!cv.canPersist) {
        throw new Error('Custom views cannot be modified in this experiment.');
      }
      // Resolve the target from a binding captured BEFORE the agent ran — never
      // from whatever view is active now, which may have changed while it was
      // running. The authoring-context latch comes first because it is the
      // truest answer: it names the view whose template the agent was actually
      // handed for this turn, and it covers requests typed straight into the
      // assistant panel, which never touch our launchers. The launch binding
      // backs it up for an empty-state build, where there is no active view for
      // the context to have described. Only a spec arriving through neither path
      // falls through to the live selection.
      const target = getDispatchedCustomViewApplyTarget() ?? pendingApplyTargetRef.current;
      const active = cv.activeView;
      const id = target?.id ?? active?.id ?? draftViewIdRef.current ?? nextViewId();
      const existingView = cv.views.find((view) => view.id === id);
      const priorView = target ?? active ?? existingView;
      // Only the validation gate is guarded. Past it the apply either landed or was
      // refused, and each of those outcomes reports itself below, so neither needs
      // the catch to classify it.
      let template: A2uiMessage[];
      try {
        template = prepareTemplate(spec);
      } catch (error) {
        throw error;
      }
      // The user-facing name comes from the launch binding when present, otherwise
      // from the active/selected view or the in-progress draft name — never from
      // the agent.
      const name = target?.name ?? active?.name ?? existingView?.name ?? cv.draftName ?? '';
      const label =
        typeof spec.title === 'string' && spec.title.trim()
          ? spec.title.trim()
          : priorView?.label || priorView?.name || name || 'Custom view';
      const createdAtMs = priorView?.createdAtMs ?? Date.now();
      // The prompt that launched a build (or the prior instruction for an
      // assistant-panel edit, whose prompt we never see) — also captured at launch.
      const instructionText = priorView?.instruction ?? '';
      const applied = cv.upsertViewContent({ id, name, label, instruction: instructionText, template, createdAtMs });
      // Released whether or not the write landed, so a refused apply can't leave
      // a stale binding to re-target the next turn.
      pendingApplyTargetRef.current = undefined;
      if (!applied) {
        // The target was deleted while the agent was running (the only refusal
        // left, since read-only experiments already threw above). Report it
        // instead of dropping it silently: deleting is deliberate, so the view
        // must stay gone, but the user still needs to know their request was
        // discarded. Throwing routes it to the inline apply error AND back to the
        // agent as a structured failure, which matters when the user is watching
        // the chat rather than this tab.
        throw new Error(
          intl.formatMessage(
            {
              defaultMessage: '"{name}" was deleted while the assistant was working, so it was not updated.',
              description:
                'Error shown when the assistant finishes building a custom view that the user deleted while the request was running',
            },
            { name: name || label },
          ),
        );
      }
      // Only track a landed id. This ref is a fallback source for `id` above, so
      // seeding it with a deleted view's id would make any later spec that falls
      // through to it resolve to a tombstone and be refused for good.
      draftViewIdRef.current = id;
    },
  });

  useEffect(() => {
    setInstruction('');
    // Staged-but-unsubmitted feedback is scoped to the trace it was entered on;
    // drop it (and the matching reset versions) when cycling so nothing leaks
    // onto a different trace's surface.
    pendingFeedbackRef.current.clear();
    feedbackResetVersionsRef.current.clear();
    bumpPendingFeedbackVersion();
  }, [traceId]);

  // Rebuild the active view's surface whenever the active view, its template, or
  // the trace changes: re-bind the stored trace-agnostic template against the
  // CURRENT trace's data. Surfaces for non-active views are torn
  // down.
  useEffect(() => {
    if (!cv.isLoaded) {
      return;
    }

    const panelsToRender = activeView ? [activeView] : [];
    const desired = new Set(panelsToRender.map(surfaceIdForView));
    for (const surfaceId of Array.from(managedSurfacesRef.current)) {
      if (!desired.has(surfaceId)) {
        processor.processMessages([{ version: 'v0.9', deleteSurface: { surfaceId } }]);
        managedSurfacesRef.current.delete(surfaceId);
        // The surface is being torn down (view switched away or deleted). Its
        // primitives' local state resets on the next mount, so drop any
        // staged-but-unsubmitted feedback for it — otherwise a later submit on
        // a rebuilt surface would log a stale value against a cleared UI.
        pendingFeedbackRef.current.delete(surfaceId);
        feedbackResetVersionsRef.current.delete(surfaceId);
        bumpPendingFeedbackVersion();
      }
    }

    for (const view of panelsToRender) {
      const surfaceId = surfaceIdForView(view);
      // Delete any prior contents so the surface rebinds cleanly to this trace.
      if (managedSurfacesRef.current.has(surfaceId)) {
        processor.processMessages([{ version: 'v0.9', deleteSurface: { surfaceId } }]);
      }

      const messages = view.unreadable
        ? placeholderMessages(
            surfaceId,
            intl.formatMessage({
              defaultMessage:
                "This view's definition couldn't be read and can't be displayed. Edit it with the assistant to rebuild it.",
              description:
                'Placeholder shown when a custom view has an invalid or unreadable definition and cannot be rendered',
            }),
          )
        : view.template && view.template.length > 0
          ? resolveTemplateForTrace(view.template, surfaceId)
          : placeholderMessages(
              surfaceId,
              intl.formatMessage({
                defaultMessage: 'This view has no content yet. Edit it with the assistant.',
                description: 'Placeholder shown for a custom view that has no rendered content yet',
              }),
            );

      processor.processMessages(messages);
      managedSurfacesRef.current.add(surfaceId);
    }

    refreshSurfaces();
  }, [activeView, cv.isLoaded, processor, resolveTemplateForTrace, intl]);

  // Opens the agent with the typed prompt as its first message; the reply's tool
  // call is applied via onSpec. Used by the empty state.
  const handleSubmitPrompt = () => {
    const prompt = instruction.trim();
    if (!cv.canPersist || !prompt || !assistant.openAssistant) {
      return;
    }
    try {
      // A brand-new view build starts a FRESH assistant session; edits (handleEditWithAssistant)
      // reuse the current session so the conversation continues.
      assistant.openAssistant(prompt, { newSession: true });
    } catch {
      // Host launcher failed synchronously — keep the typed prompt for retry and
      // don't switch into the building state for a build that never started.
      return;
    }
    // Bind the target for this build now, at launch: the empty-state box only
    // shows with no active view, so allocate a fresh id + createdAtMs and carry
    // the draft name + typed prompt. A spec that lands after the user switches
    // views then still materializes THIS new view rather than overwriting
    // whatever is active.
    const id = nextViewId();
    draftViewIdRef.current = id;
    pendingApplyTargetRef.current = {
      id,
      name: cv.draftName ?? '',
      label: undefined,
      instruction: prompt,
      createdAtMs: Date.now(),
    };
    setInstruction('');
    // Clear any leftover error from a prior failed build so the skeleton effect
    // (which cancels on applyError) doesn't immediately suppress this retry.
    assistant.clearApplyError();
    cv.startBuilding();
  };

  // Clears the building skeleton once the built view exists (success) or the
  // spec apply failed (error). Do NOT clear on the isStreaming falling edge -
  // text streaming often finishes before render_custom_view runs, so a bound
  // view may not exist yet
  useEffect(() => {
    if (!isInitialBuilding) {
      return;
    }
    if (!cv.canPersist || activeView || assistant.applyError) {
      stopBuilding();
    }
  }, [isInitialBuilding, cv.canPersist, activeView, assistant.applyError, stopBuilding]);

  // Opens the assistant to edit the active view, binding the edit to that view's
  // identity NOW so a spec that lands after a mid-edit selection change still
  // applies to the view that launched the edit. The prompt is typed inside
  // the assistant's own panel (we never see it), so we carry the view's prior
  // instruction forward rather than wiping it.
  const handleEditWithAssistant = () => {
    if (!cv.canPersist || !activeView || !assistant.openAssistant) {
      return;
    }
    pendingApplyTargetRef.current = {
      id: activeView.id,
      name: activeView.name,
      label: activeView.label,
      instruction: activeView.instruction,
      createdAtMs: activeView.createdAtMs,
    };
    assistant.openAssistant();
  };

  const handleNameConfirm = () => {
    const name = nameInput.trim();
    if (!name) {
      return;
    }
    setNameModalOpen(false);
    cv.saveActiveView(name);
  };

  const handleSave = () => {
    if (!activeView) {
      return;
    }
    // A brand-new, not-yet-persisted view is named on its first save; subsequent
    // saves of an already-persisted view persist in place with no naming prompt.
    if (!cv.isActivePersisted) {
      setNameInput('');
      setNameModalOpen(true);
      return;
    }
    cv.saveActiveView();
  };

  // Opens the rename modal, capturing the target view's id + prefilling its
  // current user-facing name so the modal stays bound to that view even if the
  // active selection changes while it is open.
  const handleOpenRename = () => {
    setRenameTargetId(cv.activeViewId);
    setRenameInput(activeView?.name ?? '');
    setRenameModalOpen(true);
  };

  const handleRenameConfirm = () => {
    const name = renameInput.trim();
    if (!renameTargetId || !name) {
      return;
    }
    setRenameModalOpen(false);
    cv.renameView(renameTargetId, name);
  };

  const handleDelete = () => {
    const id = cv.activeViewId;
    if (!id) {
      return;
    }
    cv.deleteView(id);
    setDeleteModalOpen(false);
  };

  const surfaceId = activeView ? surfaceIdForView(activeView) : '';

  const surface = activeView ? processor.model.getSurface(surfaceId) : undefined;

  // Whether the active surface has a staged entry with a submittable VALUE (a
  // rating pick or a field:"value" free-text) ready to submit for this form. The
  // form match mirrors submitStagedFeedback: only entries sharing the button's
  // `formId` count, so one form's submit never enables off another form's staged
  // feedback. A rationale on its own can't be persisted (the server requires a
  // value or an explicit error on every feedback), so a rationale-only entry
  // doesn't count as submittable here. Reads
  // `pendingFeedbackVersion` so callers re-render on every stage/clear of the
  // mutable buffer ref.
  const hasStagedFeedback = useCallback(
    (formId?: string) => {
      if (!surfaceId) {
        return false;
      }
      const surfaceBuffer = pendingFeedbackRef.current.get(surfaceId);
      if (!surfaceBuffer) {
        return false;
      }
      for (const entry of surfaceBuffer.values()) {
        if (!doesFeedbackEntryMatchForm(entry, formId)) {
          continue;
        }
        if (typeof entry.value === 'string' && entry.value.length > 0) {
          return true;
        }
      }
      return false;
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps -- pendingFeedbackVersion is the reactive signal for the mutable ref.
    [surfaceId, pendingFeedbackVersion],
  );

  // Returns the staged value/rationale for a feedback entry so a control can
  // re-seed its in-progress input when the surface rebinds on a data-only refresh
  // (which remounts the primitives). Keyed identically to handleStageFeedback.
  // Reads pendingFeedbackVersion so it re-reads the mutable buffer ref reactively.
  const getStagedFeedbackValue = useCallback(
    (
      entry: { name: string; spanId?: string; formId?: string },
      field: 'value' | 'rationale' = 'value',
    ): string | undefined => {
      if (!surfaceId) {
        return undefined;
      }
      const bufferKey = feedbackBufferKey(entry);
      return pendingFeedbackRef.current.get(surfaceId)?.get(bufferKey)?.[field];
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps -- pendingFeedbackVersion is the reactive signal for the mutable ref.
    [surfaceId, pendingFeedbackVersion],
  );

  // Increments after a feedback entry is persisted, so its radio / text control
  // clears its visible value together with the host buffer (without resetting
  // another form or span's input). Keyed identically to handleStageFeedback.
  const getFeedbackResetVersion = useCallback(
    (entry: { name: string; spanId?: string; formId?: string }): number => {
      if (!surfaceId) {
        return 0;
      }
      const bufferKey = feedbackBufferKey(entry);
      return feedbackResetVersionsRef.current.get(surfaceId)?.get(bufferKey) ?? 0;
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps -- pendingFeedbackVersion is the reactive signal for the mutable ref.
    [surfaceId, pendingFeedbackVersion],
  );

  // Shown for a view that has not been saved yet (empty user-provided name): the
  // in-progress draft and any built-but-unsaved view. Views are keyed/selected by
  // `id`, so several unsaved views sharing this label never collide.
  const untitledLabel = intl.formatMessage({
    defaultMessage: 'Untitled custom view',
    description: 'Fallback label for a custom trace view that has not been saved or named yet',
  });

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm, height: '100%', minHeight: 0 }}>
      {activeView && (
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'flex-end',
            gap: theme.spacing.sm,
            padding: theme.spacing.md,
          }}
        >
          {cv.canPersist && activeView && cv.isDirty && (
            <Button
              componentId="shared.model-trace-explorer.custom-view.save"
              type="primary"
              onClick={handleSave}
              disabled={cv.isSaving}
              loading={cv.isSaving}
            >
              <FormattedMessage
                defaultMessage="Save"
                description="Button label to save the current custom trace view"
              />
            </Button>
          )}
          {cv.canPersist && activeView && assistant.isAvailable && (
            <ModelTraceExplorerAssistantButton
              componentId="shared.model-trace-explorer.custom-view.edit-existing-view-button"
              onClick={handleEditWithAssistant}
              disabled={cv.isSaving}
            >
              <FormattedMessage
                defaultMessage="Edit with Assistant"
                description="Button label to edit the current custom trace view with the MLflow assistant"
              />
            </ModelTraceExplorerAssistantButton>
          )}
          {activeView && cv.isActivePersisted && cv.canPersist && (
            <DropdownMenu.Root>
              <DropdownMenu.Trigger asChild>
                <Button
                  componentId="shared.model-trace-explorer.custom-view.more"
                  icon={<OverflowIcon />}
                  aria-label={intl.formatMessage({
                    defaultMessage: 'More view options',
                    description: 'Accessible label for the custom view overflow menu button',
                  })}
                  disabled={cv.isSaving}
                />
              </DropdownMenu.Trigger>
              <DropdownMenu.Content align="end" minWidth={150}>
                <DropdownMenu.Item
                  componentId="shared.model-trace-explorer.custom-view.rename-view-modal-open-button"
                  onClick={handleOpenRename}
                  disabled={cv.isActivePersistedUnreadable}
                >
                  <DropdownMenu.IconWrapper>
                    <PencilIcon />
                  </DropdownMenu.IconWrapper>
                  <FormattedMessage
                    defaultMessage="Rename view"
                    description="Menu item to rename the current custom trace view"
                  />
                  {cv.isActivePersistedUnreadable && (
                    <Tooltip
                      componentId="shared.model-trace-explorer.custom-view.rename-disabled-reason"
                      side="right"
                      content={intl.formatMessage({
                        defaultMessage: 'Rebuild this invalid view with the assistant and save it to enable renaming.',
                        description:
                          'Tooltip explaining why renaming is disabled for a custom view whose saved definition is unreadable',
                      })}
                    >
                      <span
                        css={{
                          display: 'inline-flex',
                          marginLeft: 'auto',
                          paddingLeft: theme.spacing.xs,
                          color: theme.colors.textSecondary,
                          pointerEvents: 'all',
                        }}
                        onClick={(e) => e.stopPropagation()}
                      >
                        <InfoIcon aria-hidden />
                      </span>
                    </Tooltip>
                  )}
                </DropdownMenu.Item>
                <DropdownMenu.Item
                  componentId="shared.model-trace-explorer.custom-view.delete-view-modal-open-button"
                  onClick={() => setDeleteModalOpen(true)}
                >
                  <DropdownMenu.IconWrapper>
                    <TrashIcon />
                  </DropdownMenu.IconWrapper>
                  <FormattedMessage
                    defaultMessage="Delete view"
                    description="Menu item to delete the current custom trace view"
                  />
                </DropdownMenu.Item>
              </DropdownMenu.Content>
            </DropdownMenu.Root>
          )}
        </div>
      )}

      {cv.canPersist && activeView && cv.isDirty && (
        <Alert
          css={{ marginLeft: theme.spacing.md, marginRight: theme.spacing.md }}
          type="info"
          closable={false}
          componentId="shared.model-trace-explorer.custom-view.draft-indicator"
          message={intl.formatMessage({
            defaultMessage: 'Draft - This view has unsaved changes. Save it to keep them.',
            description: 'Info banner shown when the current custom trace view has unsaved changes',
          })}
        />
      )}

      {cv.saveError && (
        <Alert
          css={{ marginLeft: theme.spacing.md, marginRight: theme.spacing.md }}
          type="error"
          closable={false}
          componentId="shared.model-trace-explorer.custom-view.save-error"
          message={cv.saveError}
        />
      )}

      {assistant.applyError && (
        <Alert
          css={{ marginLeft: theme.spacing.md, marginRight: theme.spacing.md }}
          type="error"
          closable={false}
          componentId="shared.model-trace-explorer.custom-view.apply-error"
          message={intl.formatMessage(
            {
              defaultMessage: 'Assistant: {error}',
              description: 'Inline error prefix shown when the assistant fails to apply a custom view spec',
            },
            { error: assistant.applyError },
          )}
        />
      )}

      <div css={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
        {!cv.isLoaded ? (
          <div css={{ padding: theme.spacing.lg }}>
            <Typography.Text color="secondary">
              <FormattedMessage
                defaultMessage="Loading saved custom views…"
                description="Loading state shown while the experiment's saved custom views are being fetched"
              />
            </Typography.Text>
          </div>
        ) : !activeView && isInitialBuilding ? (
          <div css={{ padding: theme.spacing.lg, display: 'flex', justifyContent: 'center' }}>
            <CustomViewGeneratingSkeleton />
          </div>
        ) : !activeView && cv.views.length > 0 && (!cv.isDraft || !cv.canPersist) ? (
          // Saved views exist but none is selected yet (first load, or the active
          // view was just deleted). Show a placeholder; once the user picks a view
          // it renders here and the selection persists across trace cycling.
          <div
            css={{
              height: '100%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              padding: theme.spacing.lg,
            }}
          >
            <Empty
              title={intl.formatMessage({
                defaultMessage: 'Select a custom view',
                description: 'Empty-state title prompting the user to pick a saved custom view',
              })}
              description={intl.formatMessage({
                defaultMessage: 'Choose a saved view from the menu to render it for this trace.',
                description: 'Empty-state description prompting the user to pick a saved custom view',
              })}
            />
          </div>
        ) : !activeView && cv.views.length === 0 && !cv.canPersist ? (
          <div
            css={{
              height: '100%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              padding: theme.spacing.lg,
            }}
          >
            <Empty
              title={intl.formatMessage({
                defaultMessage: 'No custom views',
                description: 'Empty state title shown to read-only users when an experiment has no custom views',
              })}
              description={intl.formatMessage({
                defaultMessage: 'There are no saved custom views for this experiment.',
                description: 'Empty state description shown to read-only users when an experiment has no custom views',
              })}
            />
          </div>
        ) : !activeView ? (
          <div
            css={{
              height: '100%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              padding: theme.spacing.lg,
            }}
          >
            {assistant.isAvailable ? (
              <div
                css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md, width: '100%', maxWidth: 560 }}
              >
                <Typography.Title level={3} withoutMargins>
                  {intl.formatMessage({
                    defaultMessage: 'Build a custom trace view',
                    description: 'Heading for the custom view authoring empty state in the model trace explorer',
                  })}
                </Typography.Title>
                <Typography.Text color="secondary">
                  <FormattedMessage
                    defaultMessage="Describe how you want to view your trace data and the assistant will build it."
                    description="Subheading prompting the user to describe the custom trace view they want to build"
                  />
                </Typography.Text>
                <Input.TextArea
                  componentId="shared.model-trace-explorer.custom-view.create-new-view-prompt-input"
                  placeholder={intl.formatMessage({
                    defaultMessage: 'Example: Show me all the spans in this trace with their information',
                    description: 'Placeholder text for the custom trace view prompt input',
                  })}
                  value={instruction}
                  autoSize={{ minRows: 3, maxRows: 8 }}
                  onKeyDown={(event) => event.stopPropagation()}
                  onChange={(event) => setInstruction(event.target.value)}
                />
                <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
                  <ModelTraceExplorerAssistantButton
                    componentId="shared.model-trace-explorer.custom-view.build-new-custom-view-button"
                    disabled={!instruction.trim()}
                    onClick={handleSubmitPrompt}
                  >
                    <FormattedMessage
                      defaultMessage="Build with Assistant"
                      description="Button label to start building the custom trace view with the MLflow assistant"
                    />
                  </ModelTraceExplorerAssistantButton>
                  <Button
                    componentId="shared.model-trace-explorer.custom-view.cancel-new-custom-view-button"
                    type="tertiary"
                    onClick={() => setTraceExplorerDisplayMode?.('default')}
                  >
                    <Typography.Text color="secondary">
                      <FormattedMessage
                        defaultMessage="Cancel"
                        description="Button label to cancel creating a custom trace view and return to the default trace view"
                      />
                    </Typography.Text>
                  </Button>
                </div>
              </div>
            ) : (
              <Empty
                title={
                  <FormattedMessage
                    defaultMessage="Assistant unavailable"
                    description="Empty state title shown when the custom view assistant connector is not available"
                  />
                }
                description={
                  <FormattedMessage
                    defaultMessage="Custom views are built by the assistant, which isn’t available here."
                    description="Empty state description shown when the custom view assistant is not available"
                  />
                }
              />
            )}
          </div>
        ) : (
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md, padding: theme.spacing.md }}>
            <Typography.Title level={2} withoutMargins css={{ fontSize: 22, lineHeight: '28px', fontWeight: 600 }}>
              {activeView.label || untitledLabel}
            </Typography.Title>
            {surface && (
              <FeedbackStatusProvider
                value={{
                  enabled: true,
                  traceId,
                  hasStagedFeedback,
                  submitStagedFeedback,
                  getStagedFeedbackValue,
                  getFeedbackResetVersion,
                }}
              >
                <A2uiSurface key={`${surfaceId}-${traceId}`} surface={surface} />
              </FeedbackStatusProvider>
            )}
          </div>
        )}
      </div>

      <Modal
        componentId="shared.model-trace-explorer.custom-view.save-new-custom-view-with-name-modal"
        visible={nameModalOpen}
        title={intl.formatMessage({
          defaultMessage: 'Name this custom view',
          description: 'Title of the modal for naming a custom trace view on its first save',
        })}
        onCancel={() => setNameModalOpen(false)}
        onOk={handleNameConfirm}
        okText={intl.formatMessage({
          defaultMessage: 'Save',
          description: 'Confirm button label on the name-custom-view modal',
        })}
        cancelText={intl.formatMessage({
          defaultMessage: 'Cancel',
          description: 'Cancel button label on the custom view naming modal',
        })}
        okButtonProps={{ disabled: !nameInput.trim() }}
      >
        <Input
          componentId="shared.model-trace-explorer.custom-view.name-input"
          id="custom-view-name"
          aria-label={intl.formatMessage({
            defaultMessage: 'Custom view name',
            description: 'Accessible label for the custom view name input',
          })}
          value={nameInput}
          placeholder={intl.formatMessage({
            defaultMessage: 'Enter a name for this custom view',
            description: 'Placeholder text in the custom view name input',
          })}
          onChange={(event) => setNameInput(event.target.value)}
          onKeyDown={(event) => {
            event.stopPropagation();
            if (event.key === 'Enter' && nameInput.trim()) {
              handleNameConfirm();
            }
          }}
        />
      </Modal>

      <Modal
        componentId="shared.model-trace-explorer.custom-view.rename-view-modal"
        visible={renameModalOpen}
        title={intl.formatMessage({
          defaultMessage: 'Rename custom view',
          description: 'Title of the modal for renaming a custom trace view',
        })}
        onCancel={() => setRenameModalOpen(false)}
        onOk={handleRenameConfirm}
        okText={intl.formatMessage({
          defaultMessage: 'Save',
          description: 'Confirm button label on the rename-custom-view modal',
        })}
        cancelText={intl.formatMessage({
          defaultMessage: 'Cancel',
          description: 'Cancel button label on the rename-custom-view modal',
        })}
        okButtonProps={{ disabled: !renameInput.trim() }}
      >
        <Input
          componentId="shared.model-trace-explorer.custom-view.rename-input"
          id="custom-view-rename"
          aria-label={intl.formatMessage({
            defaultMessage: 'Custom view name',
            description: 'Accessible label for the custom view rename input',
          })}
          value={renameInput}
          placeholder={intl.formatMessage({
            defaultMessage: 'Enter a name for this custom view',
            description: 'Placeholder text in the custom view rename input',
          })}
          onChange={(event) => setRenameInput(event.target.value)}
          onKeyDown={(event) => {
            event.stopPropagation();
            if (event.key === 'Enter' && renameInput.trim()) {
              handleRenameConfirm();
            }
          }}
        />
      </Modal>

      <Modal
        componentId="shared.model-trace-explorer.custom-view.delete-view-modal"
        visible={deleteModalOpen}
        title={intl.formatMessage({
          defaultMessage: 'Delete view',
          description: 'Title of the delete-custom-view confirmation modal',
        })}
        onCancel={() => setDeleteModalOpen(false)}
        onOk={handleDelete}
        okText={intl.formatMessage({
          defaultMessage: 'Delete',
          description: 'Confirm button label on the delete-custom-view modal',
        })}
        cancelText={intl.formatMessage({
          defaultMessage: 'Cancel',
          description: 'Cancel button label on the delete-custom-view modal',
        })}
        okButtonProps={{ danger: true }}
      >
        <Typography.Text>
          {activeView?.name
            ? intl.formatMessage(
                {
                  defaultMessage: 'Delete the view "{name}"? This removes it from the experiment for everyone.',
                  description: 'Confirmation text when deleting a named custom trace view',
                },
                { name: activeView.name },
              )
            : intl.formatMessage({
                defaultMessage: 'Delete this view? This removes it from the experiment for everyone.',
                description: 'Confirmation text when deleting an unnamed custom trace view',
              })}
        </Typography.Text>
      </Modal>
    </div>
  );
};
