import { useMemo, useState } from 'react';

import {
  Alert,
  Button,
  ChevronLeftIcon,
  ChevronRightIcon,
  CloseSmallIcon,
  DesignSystemEventProviderAnalyticsEventTypes,
  DesignSystemEventProviderComponentTypes,
  Drawer,
  Empty,
  Input,
  TableSkeleton,
  Typography,
  useDesignSystemEventComponentCallbacks,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { ModelTraceExplorer, useGetTracesById } from '@databricks/web-shared/model-trace-explorer';
import { GenAIMarkdownRenderer } from '../../../shared/web-shared/genai-markdown-renderer';
import { FormattedMessage, useIntl } from 'react-intl';

import Utils from '../../../common/utils/Utils';
import { LabelSchemaInputRenderer } from '../../components/label-schemas';
import type { LabelSchema, LabelSchemaValue } from '../../components/label-schemas';
import { useCreateReviewAssessmentMutation } from './hooks/useCreateReviewAssessmentMutation';
import { useTraceAssessmentsQuery } from './hooks/useTraceAssessmentsQuery';
import { buildPrefilledAnswers, buildPrefilledRationales, buildPriorAssessmentIds, isAnswered } from './reviewAnswers';
import { StatusTag } from './ReviewQueueList';
import { SegmentedProgressBar } from './SegmentedProgressBar';
import type { ReviewQueueItem, ReviewStatus } from './types';

import { MARKDOWN_RENDER_SIZE_LIMIT } from '../../../shared/web-shared/model-trace-explorer/constants';

const CID = 'mlflow.experiment-review-queue.focused-review';

/** Try to parse `raw` as JSON; returns the pretty-printed string + a flag. */
const tryParseJson = (raw: string): { text: string; isJson: boolean } => {
  try {
    return { text: JSON.stringify(JSON.parse(raw), null, 2), isJson: true };
  } catch {
    return { text: raw, isJson: false };
  }
};

/**
 * Renders trace content with the same protections as the trace explorer:
 * 1. JSON → rendered as a syntax-highlighted code block (no markdown misparse)
 * 2. Content > 1 MB → plain-text fallback (prevents browser freeze)
 * 3. Everything else → markdown via GenAIMarkdownRenderer
 */
const TraceContentBubble = ({ content, variant }: { content: string; variant: 'input' | 'output' }) => {
  const { theme } = useDesignSystemTheme();
  const { text, isJson } = tryParseJson(content);

  const isInput = variant === 'input';
  const wrapperCss = isInput
    ? {
        maxWidth: '50%',
        padding: theme.spacing.sm,
        backgroundColor: theme.isDarkMode ? theme.colors.blue800 : theme.colors.blue200,
        borderRadius: theme.borders.borderRadiusMd,
        fontSize: theme.typography.fontSizeSm,
        wordBreak: 'break-word' as const,
        '& pre[class*="prism"]': { padding: `${theme.spacing.sm}px ${theme.spacing.md}px` },
      }
    : {
        padding: theme.spacing.md,
        backgroundColor: theme.isDarkMode ? theme.colors.backgroundSecondary : theme.colors.white,
        borderRadius: theme.borders.borderRadiusMd,
        fontSize: theme.typography.fontSizeSm,
        lineHeight: 1.6,
        wordBreak: 'break-word' as const,
        '& pre[class*="prism"]': { padding: `${theme.spacing.sm}px ${theme.spacing.md}px` },
      };

  const body = (() => {
    if (text.length > MARKDOWN_RENDER_SIZE_LIMIT) {
      return (
        <pre
          css={{
            margin: 0,
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-all',
            maxHeight: 400,
            overflow: 'auto',
            fontSize: theme.typography.fontSizeSm,
          }}
        >
          {text.slice(0, 10_000)}
        </pre>
      );
    }
    if (isJson) {
      return (
        <pre
          css={{
            margin: 0,
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
            fontSize: theme.typography.fontSizeSm,
            fontFamily: 'monospace',
          }}
        >
          {text}
        </pre>
      );
    }
    return <GenAIMarkdownRenderer compact={isInput}>{text}</GenAIMarkdownRenderer>;
  })();

  if (isInput) {
    return (
      <div css={{ display: 'flex', justifyContent: 'flex-end' }}>
        <div css={wrapperCss}>{body}</div>
      </div>
    );
  }
  return <div css={wrapperCss}>{body}</div>;
};

/**
 * Focused-review surface, three panels:
 *   left   — condensed queue rail (switch which trace is in focus)
 *   middle — trace summary
 *   right  — the queue's questions (label-schema widgets)
 *
 * Answering a question writes a trace assessment: a `FEEDBACK` schema
 * produces a feedback assessment, an `EXPECTATION` schema an expectation
 * assessment (see `useCreateReviewAssessmentMutation`). Marking the trace
 * `complete` / `declined` records the per-(queue, trace) shared-pool status
 * with `completedBy` attribution; reopening clears it.
 */
export const FocusedReview = ({
  item,
  items,
  schemas,
  completedBy,
  isSettingStatus,
  canReview,
  onAssignSelf,
  isAssigningSelf,
  onBack,
  onSelect,
  onSetStatus,
}: {
  item: ReviewQueueItem;
  items: ReviewQueueItem[];
  schemas: LabelSchema[];
  /** Reviewer identifier recorded on completion; `default` in no-auth OSS. */
  completedBy: string;
  /** True while a status write is in flight (disables the actions). */
  isSettingStatus: boolean;
  /** Whether the reviewer may submit reviews; when false the inputs/actions are view-only. */
  canReview: boolean;
  /** Self-assign action; shown to a manager viewing a queue they aren't in. */
  onAssignSelf?: () => void;
  isAssigningSelf?: boolean;
  onBack: () => void;
  onSelect: (itemId: string) => void;
  onSetStatus: (status: ReviewStatus) => Promise<void>;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { createReviewAssessmentAsync, isCreatingAssessment } = useCreateReviewAssessmentMutation();

  // Telemetry: count reviews (feedback) submitted from the UI. Fired after the
  // assessment writes succeed, so failed submits don't inflate the metric.
  const feedbackEvents = useMemo(() => [DesignSystemEventProviderAnalyticsEventTypes.OnClick], []);
  const feedbackSubmittedEventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.Button,
    componentId: `${CID}.feedback-submitted`,
    analyticsEvents: feedbackEvents,
  });

  // The trace's input/output for the middle panel. The full trace (spans,
  // timeline, etc.) is available on demand through the drawer.
  const { data: traceData, isLoading: traceLoading } = useGetTracesById([item.item_id]);
  const trace = traceData[0];
  const [showFullTrace, setShowFullTrace] = useState(false);
  const requestPreview = trace?.info?.request_preview;
  const responsePreview = trace?.info?.response_preview;
  const hasIO = Boolean(requestPreview || responsePreview);

  // Prefill the widgets from the trace's existing assessments; edits overlay
  // the prefill so the query result never clobbers what the reviewer typed.
  // Scope prior answers to this reviewer's own source so the prefill and the
  // supersede target are never another reviewer's answer.
  const { priorAnswers, isFetching: priorAnswersFetching } = useTraceAssessmentsQuery({
    traceId: item.item_id,
    sourceId: completedBy,
  });
  const prefilled = useMemo(() => buildPrefilledAnswers(priorAnswers, schemas), [priorAnswers, schemas]);
  const prefilledRationales = useMemo(() => buildPrefilledRationales(priorAnswers, schemas), [priorAnswers, schemas]);
  // Each question's prior assessment id (this reviewer's), so a re-submit
  // supersedes it instead of writing a duplicate.
  const priorAssessmentIds = useMemo(() => buildPriorAssessmentIds(priorAnswers, schemas), [priorAnswers, schemas]);
  const [edited, setEdited] = useState<Record<string, LabelSchemaValue>>({});
  const [editedRationales, setEditedRationales] = useState<Record<string, string>>({});
  const [submitError, setSubmitError] = useState<string | null>(null);
  // Confirmation shown after an in-place edit of a completed trace is saved;
  // cleared as soon as the reviewer edits again.
  const [editSaved, setEditSaved] = useState(false);

  const valueFor = (name: string): LabelSchemaValue => (name in edited ? edited[name] : prefilled[name]);
  const setAnswer = (name: string, value: LabelSchemaValue) => {
    setEditSaved(false);
    setEdited((prev) => ({ ...prev, [name]: value }));
  };
  const rationaleFor = (name: string): string =>
    name in editedRationales ? editedRationales[name] : (prefilledRationales[name] ?? '');
  const setRationale = (name: string, value: string) => {
    setEditSaved(false);
    setEditedRationales((prev) => ({ ...prev, [name]: value }));
  };

  // A completed trace stays editable: the reviewer can revise their answers and
  // re-save without the trace leaving the "done" bucket (mirrors the review app,
  // where there is no reopen — you just edit a completed item). A declined trace
  // has no answers to edit, so it stays locked; "Reopen" is the way back to it.
  const isComplete = item.status === 'COMPLETE';
  const isDeclined = item.status === 'DECLINED';
  const isTerminal = isComplete || isDeclined;
  // Whether the reviewer has touched any answer/rationale this session. The
  // "Save changes" path on a completed trace only writes when there's an edit,
  // so a no-op click doesn't re-supersede identical assessments. (A fresh submit
  // doesn't need this — a reopened trace can be re-completed straight from prefill.)
  const hasEdits = Object.keys(edited).length > 0 || Object.keys(editedRationales).length > 0;
  // A queue whose only question is a single Pass/Fail (no rationale) submits as
  // soon as the reviewer picks an answer — no Submit click needed.
  const autoSubmitSchema =
    schemas.length === 1 && schemas[0].input.pass_fail && !schemas[0].enable_comment ? schemas[0] : null;
  // For the auto-submit case the Submit button is redundant while the answer is
  // still empty (picking it submits). Show it again once there's a value, so a
  // reopened trace or a failed auto-submit still has an explicit re-submit path.
  const autoSubmitValue = autoSubmitSchema ? valueFor(autoSubmitSchema.name) : undefined;
  const hideSubmit = autoSubmitSchema != null && (autoSubmitValue === undefined || autoSubmitValue === null);

  // Submit writes one assessment per answered question; completing with zero
  // answers would mark the trace done while recording nothing, so the button
  // stays disabled until at least one question has a value. This reads the
  // committed answer state (edited/prefilled) and intentionally ignores the
  // auto-submit `answerOverrides`: those only flow through the auto-submit path
  // (single Pass/Fail), which submits directly and hides this button, so the two
  // never gate the same interaction.
  const answeredCount = useMemo(
    () => schemas.filter((s) => isAnswered(s.name in edited ? edited[s.name] : prefilled[s.name])).length,
    [schemas, edited, prefilled],
  );

  // Position in the queue + adjacent traces for prev/next navigation.
  const currentIndex = items.findIndex((i) => i.item_id === item.item_id);
  const prevItemId = currentIndex > 0 ? items[currentIndex - 1].item_id : undefined;
  const nextItemId =
    currentIndex !== -1 && currentIndex < items.length - 1 ? items[currentIndex + 1].item_id : undefined;

  // Progress across the queue: terminal (complete/declined) traces are reviewed.
  const reviewedCount = items.filter((i) => i.status === 'COMPLETE' || i.status === 'DECLINED').length;
  const totalCount = items.length;
  const percentage = totalCount > 0 ? Math.round((reviewedCount / totalCount) * 100) : 0;
  // One segment per trace (capped at 100, then proportional), filled blue up to
  // the reviewed count — ported from the universe review app's progress bar.
  const filledColor = theme.colors.blue600;
  const remainingColor = theme.isDarkMode ? theme.colors.blue800 : theme.colors.blue200;
  const maxSegments = 100;
  const segmentCount = Math.min(totalCount, maxSegments);
  const progressBarItems = Array.from({ length: segmentCount }, (_, index) => {
    const filled =
      totalCount <= maxSegments ? index < reviewedCount : Math.round((index / maxSegments) * 100) < percentage;
    return { color: filled ? filledColor : remainingColor };
  });

  // The next still-pending trace after the current one, wrapping around.
  const nextPendingItemId = items
    .slice(currentIndex + 1)
    .concat(items.slice(0, Math.max(currentIndex, 0)))
    .find((i) => i.item_id !== item.item_id && i.status === 'PENDING')?.item_id;

  const submitAnswersAndComplete = async (answerOverrides?: Record<string, LabelSchemaValue>) => {
    setSubmitError(null);
    // The supersede ids are derived from the prior-answers query, so submitting
    // against a still-refetching snapshot (e.g. reopen-and-resubmit before the
    // post-write refetch lands) could miss the prior and leave two live
    // assessments for one question. Wait for the query to settle first.
    if (priorAnswersFetching) {
      return;
    }
    // Auto-submit passes the just-picked value directly, since the answer state
    // set in the same click hasn't flushed yet.
    const effectiveValue = (name: string): LabelSchemaValue =>
      answerOverrides && name in answerOverrides ? answerOverrides[name] : valueFor(name);
    // Every answered question is (re)written here; an unchanged answer
    // re-supersedes its prior rather than being skipped.
    const answered = schemas.filter((s) => isAnswered(effectiveValue(s.name)));
    // Defensive: the Submit button is disabled with no answers, but never
    // record an empty completion if this is somehow reached.
    if (answered.length === 0) {
      return;
    }
    try {
      // Write the answers, then mark complete. Status is advanced only if every
      // write succeeds, so a partial failure leaves the trace pending for retry.
      await Promise.all(
        answered.map((s) =>
          createReviewAssessmentAsync({
            traceId: item.item_id,
            name: s.name,
            assessmentKind: s.type === 'EXPECTATION' ? 'expectation' : 'feedback',
            value: effectiveValue(s.name) as Exclude<LabelSchemaValue, null | undefined>,
            sourceId: completedBy,
            rationale: s.enable_comment ? rationaleFor(s.name).trim() || undefined : undefined,
            overrides: priorAssessmentIds[s.name],
          }),
        ),
      );
      // Telemetry: a review (feedback) was submitted from the UI. Covers the
      // explicit Submit, the single Pass/Fail auto-submit, and the edit-in-place
      // re-save — all of which land here after the writes succeed.
      feedbackSubmittedEventContext.onClick(undefined);
      // Editing an already-complete trace: the answers above are re-written
      // (superseding the priors), but the trace stays COMPLETE and we keep the
      // reviewer on it. Re-saving an edit must not flip it back to PENDING / the
      // to-do list, and its `completed_by`/`completed_time_ms` attribution stands.
      if (isComplete) {
        setEditSaved(true);
        return;
      }
      await onSetStatus('COMPLETE');
      // Keep the reviewer moving: jump to the next still-pending trace, or
      // return to the queue list once everything has been reviewed.
      if (nextPendingItemId) {
        onSelect(nextPendingItemId);
      } else {
        Utils.displayGlobalInfoNotification(
          intl.formatMessage({
            defaultMessage: 'All items in this queue have been reviewed!',
            description: 'Review focused view: toast shown when all queue items are reviewed',
          }),
        );
        onBack();
      }
    } catch (e) {
      setSubmitError(e instanceof Error ? e.message : String(e));
    }
  };

  const handleSetStatus = async (status: ReviewStatus) => {
    setSubmitError(null);
    try {
      await onSetStatus(status);
    } catch (e) {
      setSubmitError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md, height: '100%' }}>
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <Button componentId={`${CID}.back`} type="tertiary" icon={<CloseSmallIcon />} onClick={onBack}>
          <FormattedMessage defaultMessage="Exit review" description="Review focused view: exit review button" />
        </Button>
        <div css={{ flex: 1 }} />
        <StatusTag status={item.status} />
        {/* Progress across the queue's traces: count/percentage + segmented bar. */}
        <Typography.Text bold css={{ flexShrink: 0 }}>
          <FormattedMessage
            defaultMessage="{reviewed} of {total} reviewed ({percentage}%)"
            description="Review focused view: queue progress summary"
            values={{ reviewed: reviewedCount, total: totalCount, percentage }}
          />
        </Typography.Text>
        <SegmentedProgressBar items={progressBarItems} css={{ width: 240, height: theme.typography.fontSizeSm }} />
        <Button
          componentId={`${CID}.prev`}
          icon={<ChevronLeftIcon />}
          disabled={!prevItemId}
          aria-label={intl.formatMessage({
            defaultMessage: 'Previous trace',
            description: 'Review focused view: previous-trace button',
          })}
          onClick={() => prevItemId && onSelect(prevItemId)}
        />
        <Button
          componentId={`${CID}.next`}
          icon={<ChevronRightIcon />}
          disabled={!nextItemId}
          aria-label={intl.formatMessage({
            defaultMessage: 'Next trace',
            description: 'Review focused view: next-trace button',
          })}
          onClick={() => nextItemId && onSelect(nextItemId)}
        />
        <Button
          componentId={`${CID}.next-unreviewed`}
          disabled={!nextPendingItemId}
          onClick={() => nextPendingItemId && onSelect(nextPendingItemId)}
          endIcon={<ChevronRightIcon />}
        >
          <FormattedMessage
            defaultMessage="Next unreviewed"
            description="Review focused view: jump to the next pending trace"
          />
        </Button>
      </div>

      <div css={{ display: 'flex', gap: 48, flex: 1, minHeight: 0 }}>
        {/* Trace input / output (full trace available via the drawer) */}
        <div
          css={{
            flex: 1,
            minWidth: 0,
            backgroundColor: theme.colors.backgroundSecondary,
            borderRadius: theme.borders.borderRadiusMd,
            overflow: 'hidden',
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          <div css={{ padding: `${theme.spacing.sm}px ${theme.spacing.md}px 0 0`, flexShrink: 0, textAlign: 'right' }}>
            <Typography.Link
              componentId={`${CID}.view-full-trace`}
              disabled={!trace}
              onClick={() => setShowFullTrace(true)}
            >
              <FormattedMessage
                defaultMessage="View full trace"
                description="Review focused view: open the full trace explorer"
              />
            </Typography.Link>
          </div>
          <div
            css={{
              flex: 1,
              overflow: 'auto',
              padding: theme.spacing.md,
              display: 'flex',
              flexDirection: 'column',
              gap: theme.spacing.md,
            }}
          >
            {traceLoading ? (
              <TableSkeleton lines={8} />
            ) : !hasIO ? (
              <Empty
                description={
                  <FormattedMessage
                    defaultMessage="No input or output recorded for this trace."
                    description="Review focused view: trace panel empty state when there's no input/output"
                  />
                }
              />
            ) : (
              <>
                {requestPreview && <TraceContentBubble content={requestPreview} variant="input" />}
                {responsePreview && <TraceContentBubble content={responsePreview} variant="output" />}
              </>
            )}
          </div>
        </div>

        {/* Question widgets driven by the queue's label schemas */}
        <div
          css={{
            width: 420,
            flexShrink: 0,
            alignSelf: 'flex-start',
            maxHeight: '100%',
            display: 'flex',
            flexDirection: 'column',
            padding: theme.spacing.lg,
            backgroundColor: theme.colors.backgroundPrimary,
            border: `1px solid ${theme.colors.border}`,
            borderRadius: theme.borders.borderRadiusMd,
            boxShadow: theme.shadows.lg,
            overflow: 'hidden',
          }}
        >
          {/* Only the questions scroll. The actions below are a non-scrolling
              sibling, so they stay put (no sticky jiggle) when the list is long;
              when it's short this box shrinks to fit and the actions sit right
              under the last question. */}
          <div
            css={{
              flex: '0 1 auto',
              minHeight: 0,
              overflowY: 'auto',
              display: 'flex',
              flexDirection: 'column',
              gap: theme.spacing.lg,
            }}
          >
            {schemas.length === 0 ? (
              <Typography.Hint>
                <FormattedMessage
                  defaultMessage="No questions configured for this queue."
                  description="Review focused view: empty questions state"
                />
              </Typography.Hint>
            ) : (
              schemas.map((schema) => {
                const v = valueFor(schema.name);
                const answered = v !== undefined && v !== null && v !== '' && !(Array.isArray(v) && v.length === 0);
                return (
                  <div key={schema.schema_id} css={{ display: 'flex', gap: theme.spacing.sm }}>
                    <span
                      css={{
                        display: 'inline-block',
                        width: 8,
                        height: 8,
                        borderRadius: '50%',
                        backgroundColor: answered ? theme.colors.green600 : theme.colors.yellow600,
                        flexShrink: 0,
                        marginTop: 6,
                      }}
                    />
                    <div
                      css={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}
                    >
                      <Typography.Text bold>{schema.name}</Typography.Text>
                      {schema.instruction && !schema.input.text && (
                        <Typography.Hint css={{ marginBottom: theme.spacing.xs }}>{schema.instruction}</Typography.Hint>
                      )}
                      <LabelSchemaInputRenderer
                        input={schema.input}
                        value={valueFor(schema.name)}
                        onChange={(value) => {
                          setAnswer(schema.name, value);
                          if (
                            canReview &&
                            autoSubmitSchema?.schema_id === schema.schema_id &&
                            !isTerminal &&
                            !isCreatingAssessment &&
                            !isSettingStatus
                          ) {
                            submitAnswersAndComplete({ [schema.name]: value });
                          }
                        }}
                        disabled={isDeclined || !canReview}
                        componentId={`${CID}.question`}
                        label={schema.name}
                        instruction={schema.instruction}
                      />
                      {schema.enable_comment && (
                        <Input.TextArea
                          componentId={`${CID}.rationale`}
                          rows={2}
                          value={rationaleFor(schema.name)}
                          onChange={(e) => setRationale(schema.name, e.target.value)}
                          disabled={isDeclined || !canReview}
                          placeholder={intl.formatMessage({
                            defaultMessage: 'Rationale (optional)',
                            description: 'Review focused view: free-form rationale placeholder',
                          })}
                        />
                      )}
                    </div>
                  </div>
                );
              })
            )}
          </div>

          {/* Actions: fixed at the panel bottom (a non-scrolling sibling, so no
              jiggle), with the questions scrolling above. */}
          <div
            css={{
              flexShrink: 0,
              display: 'flex',
              flexDirection: 'column',
              gap: theme.spacing.sm,
              marginTop: theme.spacing.lg,
              paddingTop: theme.spacing.md,
              borderTop: `1px solid ${theme.colors.border}`,
            }}
          >
            {!canReview && (
              <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
                <Alert
                  componentId={`${CID}.not-assigned`}
                  type="info"
                  closable={false}
                  message={intl.formatMessage({
                    defaultMessage: "You're not assigned to this queue, so you can't submit reviews here.",
                    description: 'Review focused view: viewing a queue the reviewer is not assigned to',
                  })}
                />
                {onAssignSelf && (
                  <Button
                    componentId={`${CID}.assign-self`}
                    css={{ alignSelf: 'flex-start' }}
                    loading={isAssigningSelf}
                    onClick={onAssignSelf}
                  >
                    <FormattedMessage
                      defaultMessage="Assign myself"
                      description="Review focused view: self-assign to the queue to start reviewing"
                    />
                  </Button>
                )}
              </div>
            )}
            {editSaved && !submitError && (
              <Alert
                componentId={`${CID}.edit-saved`}
                type="success"
                closable
                onClose={() => setEditSaved(false)}
                message={intl.formatMessage({
                  defaultMessage: 'Changes saved',
                  description: 'Review focused view: confirmation that edited answers were saved',
                })}
              />
            )}
            {submitError && (
              <Alert
                componentId={`${CID}.submit-error`}
                type="error"
                closable
                onClose={() => setSubmitError(null)}
                message={intl.formatMessage(
                  {
                    defaultMessage: 'Could not save your review: {error}',
                    description: 'Review focused view: submit error alert',
                  },
                  { error: submitError },
                )}
              />
            )}
            <div css={{ display: 'flex', gap: theme.spacing.sm, justifyContent: 'flex-end' }}>
              {isTerminal ? (
                <Button
                  componentId={`${CID}.reopen`}
                  disabled={isSettingStatus || !canReview}
                  onClick={() => handleSetStatus('PENDING')}
                >
                  <FormattedMessage
                    defaultMessage="Move to Todo"
                    description="Review focused view: send a completed/declined trace back to the to-do list"
                  />
                </Button>
              ) : (
                <Button
                  componentId={`${CID}.decline`}
                  disabled={isSettingStatus || !canReview}
                  onClick={() => handleSetStatus('DECLINED')}
                >
                  <FormattedMessage defaultMessage="Decline" description="Review focused view: decline action" />
                </Button>
              )}
              {!hideSubmit && !isDeclined && (
                <Button
                  componentId={`${CID}.complete`}
                  type="primary"
                  disabled={
                    isCreatingAssessment ||
                    isSettingStatus ||
                    !canReview ||
                    answeredCount === 0 ||
                    priorAnswersFetching ||
                    (isComplete && !hasEdits)
                  }
                  loading={isCreatingAssessment || isSettingStatus}
                  onClick={() => submitAnswersAndComplete()}
                >
                  {isComplete ? (
                    <FormattedMessage
                      defaultMessage="Save changes"
                      description="Review focused view: re-save edited answers on an already-complete trace"
                    />
                  ) : (
                    <FormattedMessage
                      defaultMessage="Submit"
                      description="Review focused view: submit answers and mark the trace complete"
                    />
                  )}
                </Button>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Full trace explorer, opened from "View full trace". */}
      <Drawer.Root open={showFullTrace} onOpenChange={(open) => !open && setShowFullTrace(false)}>
        <Drawer.Content
          componentId={`${CID}.full-trace-drawer`}
          title={
            <Typography.Title level={3} withoutMargins>
              {item.item_id}
            </Typography.Title>
          }
          width="60vw"
        >
          {trace ? (
            <div css={{ height: '100%' }} onWheel={(e) => e.stopPropagation()}>
              <ModelTraceExplorer modelTrace={trace} initialActiveView="detail" />
            </div>
          ) : (
            <TableSkeleton lines={8} />
          )}
        </Drawer.Content>
      </Drawer.Root>
    </div>
  );
};
