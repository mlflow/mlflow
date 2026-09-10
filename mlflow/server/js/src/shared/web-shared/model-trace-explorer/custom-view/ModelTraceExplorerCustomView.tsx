import { type ReactNode, useCallback, useEffect, useMemo, useReducer, useRef, useState } from 'react';

import { uniqBy } from 'lodash';

import {
  Alert,
  Button,
  ChevronDownIcon,
  DropdownMenu,
  Empty,
  GenericSkeleton,
  InfoIcon,
  Input,
  Modal,
  OverflowIcon,
  PencilIcon,
  PlusIcon,
  SparkleFillIcon,
  SparkleIcon,
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
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';
import type { CreateAssessmentPayload } from '../api';
import { getUser } from '../../global-settings/getUser';
import { shouldUseTracesV4API } from '../FeatureUtils';
import { useCreateAssessment } from '../hooks/useCreateAssessment';
import { useTraceCachedActions } from '../hooks/useTraceCachedActions';
import { AssessmentBoard } from './catalog-primitives/AssessmentBoard';
import { AssessmentCard } from './catalog-primitives/AssessmentCard';
import { Card } from './catalog-primitives/Card';
import { FeedbackThumbsUpDownButtons } from './catalog-primitives/FeedbackThumbsUpDownButtons';
import { FEEDBACK_STAGED } from './catalog-primitives/feedbackActions';
import { FeedbackInputText } from './catalog-primitives/FeedbackInputText';
import { FeedbackSubmit } from './catalog-primitives/FeedbackSubmit';
import { Icon } from './catalog-primitives/Icon';
import { KeyValueViewer } from './catalog-primitives/KeyValueViewer';
import { Markdown } from './catalog-primitives/Markdown';
import { RadioGroup } from './catalog-primitives/RadioGroup';
import { StatCard } from './catalog-primitives/StatCard';
import { Text } from './catalog-primitives/Text';
import { FeedbackStatusProvider } from './FeedbackStatusContext';
import type { AgentNode } from './agent/buildAgentPrompt';
import { validateAndPrepareMessages, validateTemplate } from './agent/validateA2uiMessages';
import { resolveTemplate } from './resolveTemplate';
import { useCustomViewAssistantBridge } from './assistant/useCustomViewAssistantBridge';
import { getDispatchedCustomViewApplyTarget } from './assistant/customViewAuthoringContext';
import { CustomViewValidationError, type RenderCustomViewSpec } from './assistant/customViewSpecApplier';
import {
  CUSTOM_VIEW_CATALOG_ID,
  type CustomViewData,
  collectTraceAssessments,
  getAssessmentBoardItems,
  getMetricsFromTraceInfo,
  mapToAgentAssessments,
} from './customViewBuilders';
import { type CustomView, type CustomViewApplyTarget, MAX_CUSTOM_VIEWS_PER_EXPERIMENT } from './customViewDefinition';
import { useCustomViewDefinition } from './CustomViewDefinitionContext';

// Deterministic surface id per view so React/A2UI reuse the same surface across
// trace cycling (we rebuild the surface contents, not its identity).
const surfaceIdForView = (view: CustomView): string => `cv-${view.id}`;

let viewIdCounter = 0;
const nextViewId = (): string => `${Date.now().toString(36)}-${(viewIdCounter++).toString(36)}`;

const doesFeedbackEntryMatchForm = (entry: { formId?: string }, formId: string | undefined): boolean =>
  entry.formId === formId;

type PendingFeedbackEntry = {
  name: string;
  value?: string;
  rationale?: string;
  spanId?: string;
  formId?: string;
};

const feedbackEntryKey = ({ name, spanId, formId }: Pick<PendingFeedbackEntry, 'name' | 'spanId' | 'formId'>) =>
  `${formId ?? ''}\u0000${name}\u0000${spanId ?? ''}`;

const AssistantSparkleButton = ({
  componentId,
  onClick,
  disabled,
  children,
}: {
  componentId: string;
  onClick: () => void;
  disabled?: boolean;
  children: ReactNode;
}) => {
  const { theme } = useDesignSystemTheme();
  const [isHovered, setIsHovered] = useState(false);
  return (
    <Button
      componentId={componentId}
      icon={
        <span
          css={{
            display: 'inline-flex',
            marginRight: theme.spacing.xs,
            transition: 'transform 0.25s',
            transform: isHovered ? 'rotate(90deg)' : undefined,
          }}
        >
          {isHovered ? <SparkleFillIcon color="ai" /> : <SparkleIcon color="ai" />}
        </span>
      }
      disabled={disabled}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onClick={onClick}
    >
      {children}
    </Button>
  );
};

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
 * routes all authoring through the assistant. The empty-state box builds the
 * first view; "Edit with Assistant" reopens the assistant to change it. The
 * agent authors a trace-agnostic BOUND TEMPLATE once (via render_custom_view ->
 * onSpec); cycling traces re-binds it host-side with no further LLM call.
 */
export const ModelTraceExplorerCustomView = ({
  modelTraceInfo,
}: {
  modelTraceInfo: ModelTrace['info'];
}): JSX.Element => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const { nodeMap } = useModelTraceExplorerViewState();

  const cv = useCustomViewDefinition();

  const activeView = cv.activeView;

  // The catalog maps component type names to their React implementations: the
  // layout/content primitives plus the feedback controls a bound template can
  // reference.
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

  // The processor's action handler is created once, so route actions through a
  // ref that always points at the latest trace and mutation state.
  const actionHandlerRef = useRef<(action: A2uiClientAction) => void>(() => {});

  // Staged-but-unsubmitted feedback per surface, keyed by form + name + span so
  // controls in separate forms or on separate spans never collide.
  const pendingFeedbackRef = useRef<Map<string, Map<string, PendingFeedbackEntry>>>(new Map());
  // Successful entries increment their own reset version so the matching radio
  // and text controls clear after persistence while failed or newly edited
  // entries remain visible for retry.
  const feedbackResetVersionsRef = useRef<Map<string, Map<string, number>>>(new Map());
  const [pendingFeedbackVersion, bumpPendingFeedbackVersion] = useReducer((tick: number) => tick + 1, 0);

  // A single long-lived processor holds the state for the active view's surface.
  const [processor] = useState(
    () => new MessageProcessor([catalog], (action: A2uiClientAction) => actionHandlerRef.current(action)),
  );

  // Adapt the callback-based assessment mutation into a promise so a staged
  // form can submit dimensions one at a time and retain only failed entries.
  const createAssessmentAsync = (payload: CreateAssessmentPayload): Promise<void> =>
    new Promise<void>((resolve, reject) => {
      createAssessmentMutation(payload, {
        onSuccess: () => resolve(),
        onError: (error) => reject(error instanceof Error ? error : new Error(String(error))),
      });
    });

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
    const bufferKey = feedbackEntryKey({ formId, name, spanId });
    const previous = surfaceBuffer.get(bufferKey);
    const next: PendingFeedbackEntry = {
      ...previous,
      name,
      ...(typeof context['value'] === 'string' ? { value: context['value'] } : {}),
      ...(typeof context['rationale'] === 'string' ? { rationale: context['rationale'] } : {}),
      ...(spanId ? { spanId } : {}),
      ...(formId ? { formId } : {}),
    };
    // A data-only surface rebind can remount a control and stage the same bound
    // value again while its request is in flight. Preserve the entry identity
    // for a no-op restage so it is not mistaken for a newer user edit.
    if (
      previous &&
      previous.name === next.name &&
      previous.value === next.value &&
      previous.rationale === next.rationale &&
      previous.spanId === next.spanId &&
      previous.formId === next.formId
    ) {
      return;
    }
    surfaceBuffer.set(bufferKey, next);
    bumpPendingFeedbackVersion();
  };

  const submitStagedFeedback = async (formId?: string): Promise<{ submitted: number }> => {
    const activeSurfaceId = activeView ? surfaceIdForView(activeView) : undefined;
    const surfaceBuffer = activeSurfaceId ? pendingFeedbackRef.current.get(activeSurfaceId) : undefined;
    if (!activeSurfaceId || !surfaceBuffer || surfaceBuffer.size === 0) {
      throw new Error('There is no staged feedback to submit.');
    }

    const submittable: { key: string; entry: PendingFeedbackEntry; payload: CreateAssessmentPayload }[] = [];
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

    // React Query stores mutate callbacks on one observer, so sequential writes
    // keep each request's callbacks paired with its promise.
    let submitted = 0;
    let failed = 0;
    for (const { key, entry, payload } of submittable) {
      try {
        await createAssessmentAsync(payload);
        submitted += 1;
        // A trace/view transition can replace the surface's buffer while this
        // request is in flight, and the user can edit the same control before
        // it finishes. Clear/reset only when both the buffer and entry are still
        // the exact versions captured for this request.
        if (pendingFeedbackRef.current.get(activeSurfaceId) === surfaceBuffer && surfaceBuffer.get(key) === entry) {
          surfaceBuffer.delete(key);
          let resetVersions = feedbackResetVersionsRef.current.get(activeSurfaceId);
          if (!resetVersions) {
            resetVersions = new Map();
            feedbackResetVersionsRef.current.set(activeSurfaceId, resetVersions);
          }
          resetVersions.set(key, (resetVersions.get(key) ?? 0) + 1);
        }
      } catch {
        // Failed entries remain staged so the user can retry them.
        failed += 1;
      }
    }
    if (pendingFeedbackRef.current.get(activeSurfaceId) === surfaceBuffer && surfaceBuffer.size === 0) {
      pendingFeedbackRef.current.delete(activeSurfaceId);
    }
    bumpPendingFeedbackVersion();
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

  // Assessments shown by AssessmentBoard/AssessmentCard come from the base trace
  // MERGED with the trace-cached-actions store, so this view stays live with
  // feedback submitted elsewhere (e.g. the Assessments pane) without a refetch.
  // Merge is V4-only (the store is only written under V4).
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
  // never disagree and both update together.
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
  const [isAwaitingAssistantDispatch, setIsAwaitingAssistantDispatch] = useState(false);

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
  // background selection change (e.g. an assistant apply) can't retarget the
  // rename.
  const [renameTargetId, setRenameTargetId] = useState<string | undefined>(undefined);
  // Capture the delete target so a background selection change cannot retarget
  // the confirmation modal to another view.
  const [deleteTarget, setDeleteTarget] = useState<Pick<CustomView, 'id' | 'name'> | undefined>(undefined);

  const viewLimitReachedMessage = intl.formatMessage(
    {
      defaultMessage:
        'This experiment has reached the limit of {maxViews} custom views. Delete a view before creating a new one.',
      description:
        'Explains that no more custom trace views can be created because the per-experiment limit is reached',
    },
    { maxViews: MAX_CUSTOM_VIEWS_PER_EXPERIMENT },
  );

  const surfaceId = activeView ? surfaceIdForView(activeView) : '';
  const surface = activeView ? processor.model.getSurface(surfaceId) : undefined;

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
        if (doesFeedbackEntryMatchForm(entry, formId) && typeof entry.value === 'string' && entry.value.length > 0) {
          return true;
        }
      }
      return false;
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps -- this state tick makes the mutable ref reactive.
    [surfaceId, pendingFeedbackVersion],
  );

  const getFeedbackResetVersion = useCallback(
    (entry: Pick<PendingFeedbackEntry, 'name' | 'spanId' | 'formId'>) => {
      if (!surfaceId) {
        return 0;
      }
      return feedbackResetVersionsRef.current.get(surfaceId)?.get(feedbackEntryKey(entry)) ?? 0;
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps -- this state tick makes the mutable ref reactive.
    [surfaceId, pendingFeedbackVersion],
  );

  const getStagedFeedbackValue = useCallback(
    (entry: Pick<PendingFeedbackEntry, 'name' | 'spanId' | 'formId'>, field: 'value' | 'rationale' = 'value') => {
      if (!surfaceId) {
        return undefined;
      }
      return pendingFeedbackRef.current.get(surfaceId)?.get(feedbackEntryKey(entry))?.[field];
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps -- this state tick makes the mutable ref reactive.
    [surfaceId, pendingFeedbackVersion],
  );

  const managedSurfacesRef = useRef<Set<string>>(new Set());
  const managedSurfaceTemplateFingerprintsRef = useRef<Map<string, string>>(new Map());

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
  const hasObservedPendingDispatchRef = useRef(false);

  // The rebuild effect mutates the processor model AFTER render (creating new
  // surface objects). We render by reading the SurfaceModel out of the processor
  // model, so we force one render afterwards to re-read the current surface.
  // Intentionally NOT a rebuild-effect dependency, so it never re-runs the
  // rebuild (no loop).
  const [, refreshSurfaces] = useReducer((tick: number) => tick + 1, 0);

  // Parses + validates a raw agent spec into a stored, trace-agnostic BOUND
  // TEMPLATE (its `$source` / `$spanRef` markers preserved). Throws a
  // descriptive Error on failure. The template is re-bound per trace at render
  // time by `resolveTemplateForTrace` — no further LLM call.
  const prepareTemplate = (spec: RenderCustomViewSpec): A2uiMessage[] => {
    const result = validateTemplate(spec);
    if (!result.ok) {
      throw new CustomViewValidationError(result.error);
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
      const template = prepareTemplate(spec);
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
        // The target was deleted while the agent was running. Report it instead
        // of dropping it silently: the view must stay gone, but the user still
        // needs to know their request was discarded.
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
  const { isStreaming: isAssistantStreaming, isPending: isAssistantPending, clearApplyError } = assistant;
  const { startBuilding } = cv;

  useEffect(() => {
    setInstruction('');
    // Unsubmitted feedback belongs to the trace it was entered on.
    pendingFeedbackRef.current.clear();
    feedbackResetVersionsRef.current.clear();
    bumpPendingFeedbackVersion();
  }, [traceId]);

  // Rebuild the active view's surface whenever the active view, its template, or
  // the trace changes: re-bind the stored trace-agnostic template against the
  // CURRENT trace's data. Surfaces for non-active views are torn down.
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
        pendingFeedbackRef.current.delete(surfaceId);
        feedbackResetVersionsRef.current.delete(surfaceId);
        managedSurfaceTemplateFingerprintsRef.current.delete(surfaceId);
        bumpPendingFeedbackVersion();
      }
    }

    for (const view of panelsToRender) {
      const surfaceId = surfaceIdForView(view);
      const templateFingerprint = JSON.stringify(view.template ?? []);
      // Delete any prior contents so the surface rebinds cleanly to this trace.
      if (managedSurfacesRef.current.has(surfaceId)) {
        // A changed template can replace or regroup controls while keeping the
        // same surface id. Drop values staged against the prior definition, but
        // preserve them across ordinary data-only rebinds of the same template.
        if (managedSurfaceTemplateFingerprintsRef.current.get(surfaceId) !== templateFingerprint) {
          pendingFeedbackRef.current.delete(surfaceId);
          feedbackResetVersionsRef.current.delete(surfaceId);
          bumpPendingFeedbackVersion();
        }
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
      managedSurfaceTemplateFingerprintsRef.current.set(surfaceId, templateFingerprint);
    }

    refreshSurfaces();
  }, [activeView, cv.isLoaded, processor, resolveTemplateForTrace, intl]);

  // Opens the agent with the typed prompt as its first message; the reply's tool
  // call is applied via onSpec. Used by the empty state.
  const handleSubmitPrompt = () => {
    const prompt = instruction.trim();
    if (!cv.canPersist || !prompt || !assistant.openAssistant || assistant.isPending || isAwaitingAssistantDispatch) {
      return;
    }
    try {
      // A brand-new view build starts a FRESH assistant session; edits
      // (handleEditWithAssistant) reuse the current session so the conversation
      // continues.
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
    hasObservedPendingDispatchRef.current = false;
    setIsAwaitingAssistantDispatch(true);
  };

  // A queued automatic message may wait on provider setup or an API key. Keep
  // the prompt form visible until the Assistant actually begins streaming. If
  // the panel closes first, its queue is cleared and this launch is abandoned
  // without entering the building skeleton.
  useEffect(() => {
    if (!isAwaitingAssistantDispatch) {
      return;
    }
    if (activeView) {
      hasObservedPendingDispatchRef.current = false;
      setIsAwaitingAssistantDispatch(false);
      return;
    }
    if (isAssistantStreaming) {
      setInstruction('');
      clearApplyError();
      startBuilding();
      hasObservedPendingDispatchRef.current = false;
      setIsAwaitingAssistantDispatch(false);
      return;
    }
    if (isAssistantPending) {
      hasObservedPendingDispatchRef.current = true;
      return;
    }
    if (hasObservedPendingDispatchRef.current) {
      pendingApplyTargetRef.current = undefined;
      hasObservedPendingDispatchRef.current = false;
      setIsAwaitingAssistantDispatch(false);
    }
  }, [
    isAwaitingAssistantDispatch,
    activeView,
    isAssistantStreaming,
    isAssistantPending,
    clearApplyError,
    startBuilding,
  ]);

  // Clears the building skeleton once the built view exists (success) or the
  // spec apply failed (error). Do NOT clear on the isStreaming falling edge -
  // text streaming often finishes before render_custom_view runs, so a bound
  // view may not exist yet.
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
  // applies to the view that launched the edit. The prompt is typed inside the
  // assistant's own panel (we never see it), so we carry the view's prior
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

  // "Create view" takes the user straight into the draft authoring UI; the name
  // is collected later, on first save. startNewView sets isDraft so the authoring
  // empty state renders even when other saved views already exist.
  const handleCreateView = () => {
    if (!cv.canPersist || cv.hasReachedViewLimit) {
      return;
    }
    cv.startNewView('');
    setInstruction('');
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

  const handleOpenDelete = () => {
    if (!activeView) {
      return;
    }
    setDeleteTarget({ id: activeView.id, name: activeView.name });
  };

  const handleDelete = () => {
    if (!deleteTarget) {
      return;
    }
    cv.deleteView(deleteTarget.id);
    setDeleteTarget(undefined);
  };

  // Shown for a view that has not been saved yet (empty user-provided name): the
  // in-progress draft and any built-but-unsaved view. Views are keyed/selected by
  // `id`, so several unsaved views sharing this label never collide.
  const untitledLabel = intl.formatMessage({
    defaultMessage: 'Untitled custom view',
    description: 'Fallback label for a custom trace view that has not been saved or named yet',
  });

  // Muted "(Draft)" marker appended to a view's name in the switcher whenever it
  // has unsaved changes (a never-saved view or an edited saved one).
  const draftSuffix = (
    <Typography.Text css={{ color: theme.colors.textSecondary, marginLeft: theme.spacing.xs }}>
      {intl.formatMessage({
        defaultMessage: '(Draft)',
        description: 'Marker shown next to a custom trace view name when it has unsaved changes',
      })}
    </Typography.Text>
  );

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm, height: '100%', minHeight: 0 }}>
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: theme.spacing.sm,
          padding: theme.spacing.md,
        }}
      >
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
          {cv.views.length > 0 && (
            <DropdownMenu.Root>
              <DropdownMenu.Trigger asChild>
                <Button
                  componentId="shared.model-trace-explorer.custom-view.view-switcher"
                  endIcon={<ChevronDownIcon />}
                >
                  {activeView ? (
                    <>
                      {activeView.name || untitledLabel}
                      {cv.draftViewIds.has(activeView.id) && draftSuffix}
                    </>
                  ) : cv.isDraft && cv.canPersist ? (
                    <>
                      {untitledLabel}
                      {draftSuffix}
                    </>
                  ) : (
                    intl.formatMessage({
                      defaultMessage: 'Select a custom view',
                      description: 'Placeholder label on the custom view switcher when no view is selected',
                    })
                  )}
                </Button>
              </DropdownMenu.Trigger>
              <DropdownMenu.Content align="start" minWidth={200}>
                {cv.views.map((view) => (
                  <DropdownMenu.CheckboxItem
                    key={view.id}
                    componentId="shared.model-trace-explorer.custom-view.switch-view-item"
                    checked={view.id === cv.activeViewId}
                    // Selecting renders this view for the open trace; the selection
                    // then persists across trace cycling.
                    onClick={() => cv.selectView(view.id)}
                  >
                    <DropdownMenu.ItemIndicator />
                    {view.name || untitledLabel}
                    {cv.draftViewIds.has(view.id) && draftSuffix}
                  </DropdownMenu.CheckboxItem>
                ))}
                {cv.canPersist && assistant.isAvailable && (
                  <>
                    <DropdownMenu.Separator />
                    <DropdownMenu.Item
                      componentId="shared.model-trace-explorer.custom-view.create-view"
                      onClick={handleCreateView}
                      disabled={cv.hasReachedViewLimit}
                      disabledReason={cv.hasReachedViewLimit ? viewLimitReachedMessage : undefined}
                    >
                      <DropdownMenu.IconWrapper>
                        <PlusIcon />
                      </DropdownMenu.IconWrapper>
                      <FormattedMessage
                        defaultMessage="Create view"
                        description="Menu item to create a new custom trace view"
                      />
                    </DropdownMenu.Item>
                  </>
                )}
              </DropdownMenu.Content>
            </DropdownMenu.Root>
          )}
        </div>

        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
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
            <AssistantSparkleButton
              componentId="shared.model-trace-explorer.custom-view.edit"
              onClick={handleEditWithAssistant}
              disabled={cv.isSaving}
            >
              <FormattedMessage
                defaultMessage="Edit with Assistant"
                description="Button label to edit the current custom trace view with the MLflow Assistant"
              />
            </AssistantSparkleButton>
          )}
          {activeView && cv.isActivePersisted && (cv.canPersist || cv.canDelete) && (
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
                {cv.canPersist && (
                  <DropdownMenu.Item
                    componentId="shared.model-trace-explorer.custom-view.rename"
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
                          defaultMessage:
                            'Rebuild this invalid view with the assistant and save it to enable renaming.',
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
                )}
                {cv.canDelete && (
                  <DropdownMenu.Item
                    componentId="shared.model-trace-explorer.custom-view.delete-view-modal-open-button"
                    onClick={handleOpenDelete}
                  >
                    <DropdownMenu.IconWrapper>
                      <TrashIcon />
                    </DropdownMenu.IconWrapper>
                    <FormattedMessage
                      defaultMessage="Delete view"
                      description="Menu item to delete the current custom trace view"
                    />
                  </DropdownMenu.Item>
                )}
              </DropdownMenu.Content>
            </DropdownMenu.Root>
          )}
        </div>
      </div>

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
          // Saved views exist but none is selected yet (first load). Show a
          // placeholder; once the user picks a view it renders here and the
          // selection persists across trace cycling.
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
                  componentId="shared.model-trace-explorer.custom-view.prompt"
                  placeholder={intl.formatMessage({
                    defaultMessage: 'Example: Show me all the spans in this trace with their information',
                    description: 'Placeholder text for the custom trace view prompt input',
                  })}
                  value={instruction}
                  autoSize={{ minRows: 3, maxRows: 8 }}
                  onKeyDown={(event) => {
                    event.stopPropagation();
                    // Enter submits the prompt; Shift+Enter inserts a newline
                    // (matching the main assistant chat panel). Skip submit while
                    // an IME composition is being confirmed.
                    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
                      event.preventDefault();
                      handleSubmitPrompt();
                    }
                  }}
                  onChange={(event) => setInstruction(event.target.value)}
                />
                <div>
                  <AssistantSparkleButton
                    componentId="shared.model-trace-explorer.custom-view.build"
                    disabled={
                      !instruction.trim() || assistant.isPending || assistant.isStreaming || isAwaitingAssistantDispatch
                    }
                    onClick={handleSubmitPrompt}
                  >
                    <FormattedMessage
                      defaultMessage="Build with Assistant"
                      description="Button label to start building the custom trace view with the MLflow Assistant"
                    />
                  </AssistantSparkleButton>
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
                    defaultMessage="Custom views are built by the assistant, which isn't available here."
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
                  getStagedFeedbackValue,
                  submitStagedFeedback,
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
        componentId="shared.model-trace-explorer.custom-view.name-modal"
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
        componentId="shared.model-trace-explorer.custom-view.rename-modal"
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
        visible={Boolean(deleteTarget)}
        title={intl.formatMessage({
          defaultMessage: 'Delete view',
          description: 'Title of the delete-custom-view confirmation modal',
        })}
        onCancel={() => setDeleteTarget(undefined)}
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
          {deleteTarget?.name
            ? intl.formatMessage(
                {
                  defaultMessage: 'Delete the view "{name}"? This removes it from the experiment for everyone.',
                  description: 'Confirmation text when deleting a named custom trace view',
                },
                { name: deleteTarget.name },
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
