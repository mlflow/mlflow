import { Button, ChevronLeftIcon, Input, Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';

import type { ReviewItem, ReviewQuestion, ReviewStatus } from './mockData';
import { StatusTag } from './ReviewQueueList';

const CID = 'mlflow.experiment-review-queue.focused-review';

/**
 * POC focused-review surface: trace render (mocked) on the left,
 * schema-defined question widgets on the right, with the
 * mark-complete / skip / reopen actions.
 */
export const FocusedReview = ({
  item,
  questions,
  onBack,
  onUpdate,
  onSetStatus,
}: {
  item: ReviewItem;
  questions: ReviewQuestion[];
  onBack: () => void;
  onUpdate: (patch: Partial<ReviewItem>) => void;
  onSetStatus: (status: ReviewStatus) => void;
}) => {
  const { theme } = useDesignSystemTheme();

  const setAnswer = (name: string, value: string | number) => onUpdate({ answers: { ...item.answers, [name]: value } });

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md, height: '100%' }}>
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <Button componentId={`${CID}.back`} icon={<ChevronLeftIcon />} onClick={onBack}>
          Back to queue
        </Button>
        <Typography.Text bold>{item.traceId}</Typography.Text>
        <StatusTag status={item.status} />
      </div>

      <div css={{ display: 'flex', gap: theme.spacing.lg, flex: 1, minHeight: 0 }}>
        {/* Trace render (mocked) */}
        <div
          css={{
            flex: 1,
            border: `1px solid ${theme.colors.border}`,
            borderRadius: theme.borders.borderRadiusMd,
            padding: theme.spacing.md,
            overflow: 'auto',
          }}
        >
          <Typography.Title level={4}>Trace</Typography.Title>
          <Typography.Text color="secondary">Request</Typography.Text>
          <div css={{ marginBottom: theme.spacing.md }}>
            <Typography.Paragraph>{item.requestPreview}</Typography.Paragraph>
          </div>
          <Typography.Text color="secondary">Response</Typography.Text>
          <Typography.Paragraph>{item.responsePreview}</Typography.Paragraph>
        </div>

        {/* Question widgets */}
        <div
          css={{
            width: 360,
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.lg,
            borderLeft: `1px solid ${theme.colors.border}`,
            paddingLeft: theme.spacing.lg,
            overflow: 'auto',
          }}
        >
          <Typography.Title level={4} withoutMargins>
            Review
          </Typography.Title>

          {questions.map((q) => (
            <div key={q.name} css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
              <Typography.Text bold>{q.title}</Typography.Text>
              {q.instruction && (
                <Typography.Hint css={{ marginBottom: theme.spacing.xs }}>{q.instruction}</Typography.Hint>
              )}
              <QuestionWidget question={q} value={item.answers[q.name]} onChange={(v) => setAnswer(q.name, v)} />
            </div>
          ))}

          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
            <Typography.Text bold>Comment</Typography.Text>
            <Input.TextArea
              componentId={`${CID}.comment`}
              value={item.comment ?? ''}
              onChange={(e) => onUpdate({ comment: e.target.value })}
              autoSize={{ minRows: 2, maxRows: 6 }}
              placeholder="Optional free-form feedback"
            />
          </div>
        </div>
      </div>

      {/* Actions */}
      <div
        css={{
          display: 'flex',
          gap: theme.spacing.sm,
          justifyContent: 'flex-end',
          borderTop: `1px solid ${theme.colors.border}`,
          paddingTop: theme.spacing.md,
        }}
      >
        {item.status === 'COMPLETED' || item.status === 'SKIPPED' ? (
          <Button componentId={`${CID}.reopen`} onClick={() => onSetStatus('PENDING')}>
            Reopen
          </Button>
        ) : (
          <Button componentId={`${CID}.skip`} onClick={() => onSetStatus('SKIPPED')}>
            Skip
          </Button>
        )}
        <Button
          componentId={`${CID}.complete`}
          type="primary"
          disabled={item.status === 'COMPLETED'}
          onClick={() => onSetStatus('COMPLETED')}
        >
          Mark complete
        </Button>
      </div>
    </div>
  );
};

const QuestionWidget = ({
  question,
  value,
  onChange,
}: {
  question: ReviewQuestion;
  value: string | number | undefined;
  onChange: (value: string | number) => void;
}) => {
  const { theme } = useDesignSystemTheme();

  if (question.type === 'numeric') {
    return (
      <Input
        componentId={`${CID}.question-numeric`}
        type="number"
        value={value === undefined ? '' : String(value)}
        onChange={(e) => onChange(e.target.value === '' ? '' : Number(e.target.value))}
        css={{ width: 120 }}
      />
    );
  }

  const options = question.type === 'pass_fail' ? ['Pass', 'Fail'] : (question.options ?? []);
  return (
    <div css={{ display: 'flex', flexWrap: 'wrap', gap: theme.spacing.xs }}>
      {options.map((opt) => (
        <Button
          key={opt}
          componentId={`${CID}.question-option`}
          size="small"
          type={value === opt ? 'primary' : undefined}
          onClick={() => onChange(opt)}
        >
          {opt}
        </Button>
      ))}
    </div>
  );
};
