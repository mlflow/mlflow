import { useMemo, useState } from 'react';

import {
  Alert,
  Button,
  ChevronLeftIcon,
  ChevronRightIcon,
  Drawer,
  Empty,
  Input,
  TableSkeleton,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { ModelTraceExplorer, useGetTracesById } from '@databricks/web-shared/model-trace-explorer';
import { FormattedMessage, useIntl } from 'react-intl';

import { LabelSchemaInputRenderer } from '../../components/label-schemas';
import type { LabelSchema, LabelSchemaValue } from '../../components/label-schemas';
import { useCreateReviewAssessmentMutation } from './hooks/useCreateReviewAssessmentMutation';
import { useTraceAssessmentsQuery } from './hooks/useTraceAssessmentsQuery';
import { buildPrefilledAnswers, buildPrefilledRationales } from './reviewAnswers';
import { StatusTag } from './ReviewQueueList';
import type { ReviewQueueItem, ReviewStatus } from './types';

const CID = 'mlflow.experiment-review-queue.focused-review';

/** Pretty-print a request/response preview as JSON, falling back to the raw string. */
const formatPreview = (raw: string): string => {
  try {
    return JSON.stringify(JSON.parse(raw), null, 2);
  } catch {
    return raw;
  }
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
  onBack: () => void;
  onSelect: (targetId: string) => void;
  onSetStatus: (status: ReviewStatus) => Promise<void>;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { createReviewAssessmentAsync, isCreatingAssessment } = useCreateReviewAssessmentMutation();

  // The trace's input/output for the middle panel. The full trace (spans,
  // timeline, etc.) is available on demand through the drawer.
  const { data: traceData, isLoading: traceLoading } = useGetTracesById([item.target_id]);
  const trace = traceData[0];
  const [showFullTrace, setShowFullTrace] = useState(false);
  const requestPreview = trace?.info?.request_preview;
  const responsePreview = trace?.info?.response_preview;
  const hasIO = Boolean(requestPreview || responsePreview);

  // Prefill the widgets from the trace's existing assessments; edits overlay
  // the prefill so the query result never clobbers what the reviewer typed.
  const { priorAnswers } = useTraceAssessmentsQuery({ traceId: item.target_id });
  const prefilled = useMemo(() => buildPrefilledAnswers(priorAnswers, schemas), [priorAnswers, schemas]);
  const prefilledRationales = useMemo(() => buildPrefilledRationales(priorAnswers, schemas), [priorAnswers, schemas]);
  const [edited, setEdited] = useState<Record<string, LabelSchemaValue>>({});
  const [editedRationales, setEditedRationales] = useState<Record<string, string>>({});
  const [submitError, setSubmitError] = useState<string | null>(null);

  const valueFor = (name: string): LabelSchemaValue => (name in edited ? edited[name] : prefilled[name]);
  const setAnswer = (name: string, value: LabelSchemaValue) => setEdited((prev) => ({ ...prev, [name]: value }));
  const rationaleFor = (name: string): string =>
    name in editedRationales ? editedRationales[name] : (prefilledRationales[name] ?? '');
  const setRationale = (name: string, value: string) => setEditedRationales((prev) => ({ ...prev, [name]: value }));

  const isTerminal = item.status === 'COMPLETE' || item.status === 'DECLINED';

  // Position in the queue + adjacent traces for prev/next navigation.
  const currentIndex = items.findIndex((i) => i.target_id === item.target_id);
  const prevTargetId = currentIndex > 0 ? items[currentIndex - 1].target_id : undefined;
  const nextTargetId =
    currentIndex !== -1 && currentIndex < items.length - 1 ? items[currentIndex + 1].target_id : undefined;

  // Progress across the queue: terminal (complete/declined) traces are reviewed.
  const reviewedCount = items.filter((i) => i.status === 'COMPLETE' || i.status === 'DECLINED').length;
  const totalCount = items.length;
  const percentage = totalCount > 0 ? Math.round((reviewedCount / totalCount) * 100) : 0;

  // The next still-pending trace after the current one, wrapping around.
  const nextPendingTargetId = items
    .slice(currentIndex + 1)
    .concat(items.slice(0, Math.max(currentIndex, 0)))
    .find((i) => i.target_id !== item.target_id && i.status === 'PENDING')?.target_id;

  const submitAnswersAndComplete = async () => {
    setSubmitError(null);
    const answered = schemas.filter((s) => {
      const v = valueFor(s.name);
      return v !== undefined && v !== null && v !== '' && !(Array.isArray(v) && v.length === 0);
    });
    try {
      // Write the answers, then mark complete. Status is advanced only if every
      // write succeeds, so a partial failure leaves the trace pending for retry.
      await Promise.all(
        answered.map((s) =>
          createReviewAssessmentAsync({
            traceId: item.target_id,
            name: s.name,
            assessmentKind: s.type === 'EXPECTATION' ? 'expectation' : 'feedback',
            value: valueFor(s.name) as Exclude<LabelSchemaValue, null | undefined>,
            sourceId: completedBy,
            rationale: s.enable_comment ? rationaleFor(s.name).trim() || undefined : undefined,
          }),
        ),
      );
      await onSetStatus('COMPLETE');
      // Keep the reviewer moving: jump to the next still-pending trace, or
      // return to the queue list once everything has been reviewed.
      if (nextPendingTargetId) {
        onSelect(nextPendingTargetId);
      } else {
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
        <Button componentId={`${CID}.back`} icon={<ChevronLeftIcon />} onClick={onBack}>
          <FormattedMessage defaultMessage="Back to traces" description="Review focused view: back button" />
        </Button>
        <Typography.Text bold>{item.target_id}</Typography.Text>
        <StatusTag status={item.status} />
        <div css={{ flex: 1 }} />
        <Button
          componentId={`${CID}.prev`}
          icon={<ChevronLeftIcon />}
          disabled={!prevTargetId}
          aria-label={intl.formatMessage({
            defaultMessage: 'Previous trace',
            description: 'Review focused view: previous-trace button',
          })}
          onClick={() => prevTargetId && onSelect(prevTargetId)}
        />
        <Button
          componentId={`${CID}.next`}
          icon={<ChevronRightIcon />}
          disabled={!nextTargetId}
          aria-label={intl.formatMessage({
            defaultMessage: 'Next trace',
            description: 'Review focused view: next-trace button',
          })}
          onClick={() => nextTargetId && onSelect(nextTargetId)}
        />
      </div>

      {/* Progress across the queue's traces */}
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
        <Typography.Text bold size="lg">
          <FormattedMessage
            defaultMessage="{reviewed} of {total} reviewed ({percentage}%)"
            description="Review focused view: queue progress summary"
            values={{ reviewed: reviewedCount, total: totalCount, percentage }}
          />
        </Typography.Text>
        <div
          css={{
            width: '100%',
            height: theme.spacing.sm,
            borderRadius: theme.borders.borderRadiusMd,
            backgroundColor: theme.colors.backgroundSecondary,
            overflow: 'hidden',
          }}
        >
          <div
            css={{
              width: `${percentage}%`,
              height: '100%',
              backgroundColor: theme.colors.primary,
              transition: 'width 0.2s ease',
            }}
          />
        </div>
      </div>

      <div css={{ display: 'flex', gap: theme.spacing.lg, flex: 1, minHeight: 0 }}>
        {/* Trace input / output (full trace available via the drawer) */}
        <div
          css={{
            flex: 1,
            minWidth: 0,
            border: `1px solid ${theme.colors.border}`,
            borderRadius: theme.borders.borderRadiusMd,
            overflow: 'hidden',
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          <div
            css={{
              display: 'flex',
              alignItems: 'center',
              gap: theme.spacing.sm,
              padding: theme.spacing.sm,
              borderBottom: `1px solid ${theme.colors.border}`,
            }}
          >
            <Typography.Title level={4} withoutMargins>
              <FormattedMessage defaultMessage="Trace" description="Review focused view: trace panel title" />
            </Typography.Title>
            <div css={{ flex: 1 }} />
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
              [
                requestPreview && {
                  key: 'input',
                  label: (
                    <FormattedMessage defaultMessage="Input" description="Review focused view: trace input label" />
                  ),
                  value: formatPreview(requestPreview),
                },
                responsePreview && {
                  key: 'output',
                  label: (
                    <FormattedMessage defaultMessage="Output" description="Review focused view: trace output label" />
                  ),
                  value: formatPreview(responsePreview),
                },
              ]
                .filter((section): section is { key: string; label: JSX.Element; value: string } => Boolean(section))
                .map((section) => (
                  <div key={section.key} css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                    <Typography.Text bold>{section.label}</Typography.Text>
                    <pre
                      css={{
                        margin: 0,
                        padding: theme.spacing.sm,
                        backgroundColor: theme.colors.backgroundSecondary,
                        borderRadius: theme.borders.borderRadiusMd,
                        fontFamily: 'monospace',
                        fontSize: theme.typography.fontSizeSm,
                        whiteSpace: 'pre-wrap',
                        wordBreak: 'break-word',
                      }}
                    >
                      {section.value}
                    </pre>
                  </div>
                ))
            )}
          </div>
        </div>

        {/* Question widgets driven by the queue's label schemas */}
        <div
          css={{
            width: 360,
            flexShrink: 0,
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.lg,
            borderLeft: `1px solid ${theme.colors.border}`,
            paddingLeft: theme.spacing.lg,
            overflow: 'auto',
          }}
        >
          <Typography.Title level={4} withoutMargins>
            <FormattedMessage defaultMessage="Review" description="Review focused view: questions panel title" />
          </Typography.Title>

          {schemas.length === 0 ? (
            <Typography.Hint>
              <FormattedMessage
                defaultMessage="No questions configured for this queue."
                description="Review focused view: empty questions state"
              />
            </Typography.Hint>
          ) : (
            schemas.map((schema) => (
              <div key={schema.schema_id} css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                <Typography.Text bold>{schema.name}</Typography.Text>
                {schema.instruction && (
                  <Typography.Hint css={{ marginBottom: theme.spacing.xs }}>{schema.instruction}</Typography.Hint>
                )}
                <LabelSchemaInputRenderer
                  input={schema.input}
                  value={valueFor(schema.name)}
                  onChange={(value) => setAnswer(schema.name, value)}
                  disabled={isTerminal}
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
                    disabled={isTerminal}
                    placeholder={intl.formatMessage({
                      defaultMessage: 'Rationale (optional)',
                      description: 'Review focused view: free-form rationale placeholder',
                    })}
                  />
                )}
              </div>
            ))
          )}
        </div>
      </div>

      {/* Actions */}
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.sm,
          borderTop: `1px solid ${theme.colors.border}`,
          paddingTop: theme.spacing.md,
        }}
      >
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
            <Button componentId={`${CID}.reopen`} disabled={isSettingStatus} onClick={() => handleSetStatus('PENDING')}>
              <FormattedMessage defaultMessage="Reopen" description="Review focused view: reopen action" />
            </Button>
          ) : (
            <Button
              componentId={`${CID}.decline`}
              disabled={isSettingStatus}
              onClick={() => handleSetStatus('DECLINED')}
            >
              <FormattedMessage defaultMessage="Decline" description="Review focused view: decline action" />
            </Button>
          )}
          <Button
            componentId={`${CID}.complete`}
            type="primary"
            disabled={isTerminal || isCreatingAssessment || isSettingStatus}
            loading={isCreatingAssessment || isSettingStatus}
            onClick={submitAnswersAndComplete}
          >
            <FormattedMessage
              defaultMessage="Submit"
              description="Review focused view: submit answers and mark the trace complete"
            />
          </Button>
        </div>
      </div>

      {/* Full trace explorer, opened from "View full trace". */}
      <Drawer.Root open={showFullTrace} onOpenChange={(open) => !open && setShowFullTrace(false)}>
        <Drawer.Content
          componentId={`${CID}.full-trace-drawer`}
          title={
            <Typography.Title level={3} withoutMargins>
              {item.target_id}
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
