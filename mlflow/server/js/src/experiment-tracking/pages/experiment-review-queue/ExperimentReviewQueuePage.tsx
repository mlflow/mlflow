import { useMemo, useState } from 'react';

import { Button, GearIcon, Tag, Tooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';

import { FocusedReview } from './FocusedReview';
import { ManageQuestionsModal } from './ManageQuestionsModal';
import { ReviewQueueList } from './ReviewQueueList';
import {
  MOCK_QUEUES,
  MOCK_QUESTIONS,
  MOCK_REVIEWERS,
  type ReviewItem,
  type ReviewQuestion,
  type ReviewStatus,
} from './mockData';

const CID = 'mlflow.experiment-review-queue.page';

/** Deep-copy the mock fixture so in-session edits don't mutate it. */
const cloneQueues = (queues: Record<string, ReviewItem[]>): Record<string, ReviewItem[]> =>
  Object.fromEntries(
    Object.entries(queues).map(([reviewer, items]) => [
      reviewer,
      items.map((i) => ({ ...i, answers: { ...i.answers } })),
    ]),
  );

/**
 * Review Queue tab — proof of concept.
 *
 * Renders entirely off in-memory mock data ({@link MOCK_QUEUES}) so we can
 * walk the reviewer flows (open queue → review a trace → mark complete /
 * skip / reopen) and gather feedback before committing to the eng design.
 * Nothing here talks to a backend. The "Viewing as" switcher stands in for
 * the logged-in reviewer to demonstrate that each reviewer has their own
 * queue.
 */
const ExperimentReviewQueuePage = () => {
  const { theme } = useDesignSystemTheme();
  const [queues, setQueues] = useState<Record<string, ReviewItem[]>>(() => cloneQueues(MOCK_QUEUES));
  const [reviewerId, setReviewerId] = useState(MOCK_REVIEWERS[0].id);
  const [openAssignmentId, setOpenAssignmentId] = useState<string | null>(null);
  const [questions, setQuestions] = useState<ReviewQuestion[]>(MOCK_QUESTIONS);
  const [manageOpen, setManageOpen] = useState(false);

  const nowMs = Date.now();
  const items = useMemo(() => queues[reviewerId] ?? [], [queues, reviewerId]);
  const openItem = useMemo(
    () => items.find((i) => i.assignmentId === openAssignmentId) ?? null,
    [items, openAssignmentId],
  );

  const patchOpenItem = (patch: Partial<ReviewItem>) => {
    if (!openAssignmentId) {
      return;
    }
    setQueues((prev) => ({
      ...prev,
      [reviewerId]: prev[reviewerId].map((i) => (i.assignmentId === openAssignmentId ? { ...i, ...patch } : i)),
    }));
  };

  const setOpenStatus = (status: ReviewStatus) => {
    // Stay in the focused view after a terminal action; the reviewer
    // picks the next trace from the condensed queue rail. "Back to
    // queue" is the explicit exit to the full list.
    patchOpenItem({ status });
  };

  const pendingCount = items.filter((i) => i.status === 'PENDING').length;

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.md,
        height: '100%',
        padding: theme.spacing.md,
      }}
    >
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <Typography.Title level={2} withoutMargins>
          Review
        </Typography.Title>
        <Tooltip componentId={`${CID}.edit-questions-tooltip`} content="Edit review questions for this experiment">
          <Button
            componentId={`${CID}.edit-questions`}
            icon={<GearIcon />}
            aria-label="Edit review questions"
            onClick={() => setManageOpen(true)}
          />
        </Tooltip>
        <Tag componentId={`${CID}.poc-tag`} color="lemon">
          Proof of concept · dummy data
        </Tag>
      </div>

      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <Typography.Text color="secondary">Viewing as:</Typography.Text>
        {MOCK_REVIEWERS.map((r) => (
          <Button
            key={r.id}
            componentId={`${CID}.switch-reviewer`}
            size="small"
            type={r.id === reviewerId ? 'primary' : undefined}
            onClick={() => {
              setReviewerId(r.id);
              setOpenAssignmentId(null);
            }}
          >
            {r.displayName}
          </Button>
        ))}
        {!openItem && (
          <Typography.Hint css={{ marginLeft: 'auto' }}>
            {pendingCount} {pendingCount === 1 ? 'trace' : 'traces'} left to review
          </Typography.Hint>
        )}
      </div>

      <div css={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
        {openItem ? (
          <FocusedReview
            item={openItem}
            items={items}
            questions={questions}
            onBack={() => setOpenAssignmentId(null)}
            onSelect={setOpenAssignmentId}
            onUpdate={patchOpenItem}
            onSetStatus={setOpenStatus}
          />
        ) : (
          <ReviewQueueList items={items} onOpen={(item) => setOpenAssignmentId(item.assignmentId)} nowMs={nowMs} />
        )}
      </div>

      {manageOpen && (
        <ManageQuestionsModal questions={questions} onChange={setQuestions} onClose={() => setManageOpen(false)} />
      )}
    </div>
  );
};

export default ExperimentReviewQueuePage;
