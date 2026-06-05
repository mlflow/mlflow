import { useMemo, useState } from 'react';

import { Alert, Button, ChevronLeftIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import { LabelSchemaInputRenderer } from '../../components/label-schemas';
import type { LabelSchema, LabelSchemaValue } from '../../components/label-schemas';
import { useCreateReviewAssessmentMutation } from './hooks/useCreateReviewAssessmentMutation';
import { useTraceAssessmentsQuery } from './hooks/useTraceAssessmentsQuery';
import { buildPrefilledAnswers } from './reviewAnswers';
import { StatusTag } from './ReviewQueueList';
import type { ReviewQueueItem, ReviewStatus } from './types';

const CID = 'mlflow.experiment-review-queue.focused-review';

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

  // Prefill the widgets from the trace's existing assessments; edits overlay
  // the prefill so the query result never clobbers what the reviewer typed.
  const { priorAnswers } = useTraceAssessmentsQuery({ traceId: item.target_id });
  const prefilled = useMemo(() => buildPrefilledAnswers(priorAnswers, schemas), [priorAnswers, schemas]);
  const [edited, setEdited] = useState<Record<string, LabelSchemaValue>>({});
  const [submitError, setSubmitError] = useState<string | null>(null);

  const valueFor = (name: string): LabelSchemaValue => (name in edited ? edited[name] : prefilled[name]);
  const setAnswer = (name: string, value: LabelSchemaValue) => setEdited((prev) => ({ ...prev, [name]: value }));

  const isTerminal = item.status === 'COMPLETE' || item.status === 'DECLINED';

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
          }),
        ),
      );
      await onSetStatus('COMPLETE');
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
          <FormattedMessage defaultMessage="Back to queue" description="Review focused view: back button" />
        </Button>
        <Typography.Text bold>{item.target_id}</Typography.Text>
        <StatusTag status={item.status} />
      </div>

      <div css={{ display: 'flex', gap: theme.spacing.lg, flex: 1, minHeight: 0 }}>
        {/* Condensed queue rail */}
        <div
          css={{
            width: 240,
            flexShrink: 0,
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.xs,
            borderRight: `1px solid ${theme.colors.border}`,
            paddingRight: theme.spacing.md,
            overflow: 'auto',
          }}
        >
          <Typography.Text color="secondary" size="sm" css={{ marginBottom: theme.spacing.xs }}>
            <FormattedMessage
              defaultMessage="Queue ({count})"
              description="Review focused view: queue rail header with trace count"
              values={{ count: items.length }}
            />
          </Typography.Text>
          {items.map((queued) => {
            const isActive = queued.target_id === item.target_id;
            return (
              <div
                key={queued.target_id}
                role="button"
                tabIndex={0}
                onClick={() => onSelect(queued.target_id)}
                onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && onSelect(queued.target_id)}
                css={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: theme.spacing.xs,
                  padding: theme.spacing.sm,
                  borderRadius: theme.borders.borderRadiusMd,
                  cursor: 'pointer',
                  backgroundColor: isActive ? theme.colors.actionDefaultBackgroundPress : undefined,
                  '&:hover': { backgroundColor: theme.colors.actionDefaultBackgroundHover },
                }}
              >
                <Typography.Text size="sm" bold={isActive} ellipsis>
                  {queued.target_id}
                </Typography.Text>
                <StatusTag status={queued.status} />
              </div>
            );
          })}
        </div>

        {/* Trace summary */}
        <div
          css={{
            flex: 1,
            minWidth: 0,
            border: `1px solid ${theme.colors.border}`,
            borderRadius: theme.borders.borderRadiusMd,
            padding: theme.spacing.md,
            overflow: 'auto',
          }}
        >
          <Typography.Title level={4}>
            <FormattedMessage defaultMessage="Trace" description="Review focused view: trace panel title" />
          </Typography.Title>
          <Typography.Text>{item.target_id}</Typography.Text>
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
              defaultMessage="Submit & mark complete"
              description="Review focused view: submit answers and mark the trace complete"
            />
          </Button>
        </div>
      </div>
    </div>
  );
};
