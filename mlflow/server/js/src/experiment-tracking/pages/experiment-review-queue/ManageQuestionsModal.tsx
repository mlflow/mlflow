import { useState } from 'react';

import { Button, Input, Modal, Tag, TrashIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';

import type { QuestionType, ReviewQuestion } from './mockData';

const CID = 'mlflow.experiment-review-queue.manage-questions';

const TYPE_OPTIONS: { type: QuestionType; label: string }[] = [
  { type: 'pass_fail', label: 'Pass / Fail' },
  { type: 'categorical', label: 'Categorical' },
  { type: 'numeric', label: 'Numeric' },
];

const TYPE_LABEL: Record<QuestionType, string> = {
  pass_fail: 'Pass / Fail',
  categorical: 'Categorical',
  numeric: 'Numeric',
};

const slugify = (title: string) =>
  title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_|_$/g, '') || 'question';

/**
 * POC "edit the questions for this experiment" surface, opened from the
 * gear icon on the Review page. Mirrors (in spirit) the DAIS-2 label
 * schema admin/create form — add, retitle, and remove the schema-style
 * questions reviewers answer. In-memory only.
 */
export const ManageQuestionsModal = ({
  questions,
  onChange,
  onClose,
}: {
  questions: ReviewQuestion[];
  onChange: (questions: ReviewQuestion[]) => void;
  onClose: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const [newTitle, setNewTitle] = useState('');
  const [newType, setNewType] = useState<QuestionType>('pass_fail');
  const [newOptions, setNewOptions] = useState('');

  const retitle = (name: string, title: string) =>
    onChange(questions.map((q) => (q.name === name ? { ...q, title } : q)));

  const remove = (name: string) => onChange(questions.filter((q) => q.name !== name));

  const addQuestion = () => {
    const title = newTitle.trim();
    if (!title) {
      return;
    }
    const options =
      newType === 'categorical'
        ? newOptions
            .split(',')
            .map((o) => o.trim())
            .filter(Boolean)
        : undefined;
    const question: ReviewQuestion = {
      name: `${slugify(title)}_${questions.length}`,
      title,
      type: newType,
      ...(options && options.length > 0 ? { options } : {}),
    };
    onChange([...questions, question]);
    setNewTitle('');
    setNewOptions('');
    setNewType('pass_fail');
  };

  const canAdd = newTitle.trim().length > 0 && (newType !== 'categorical' || newOptions.trim().length > 0);

  return (
    <Modal
      visible
      componentId={CID}
      title="Edit review questions"
      onCancel={onClose}
      footer={
        <Button componentId={`${CID}.done`} type="primary" onClick={onClose}>
          Done
        </Button>
      }
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        <Typography.Text color="secondary">
          Questions are defined once per experiment and apply to every assigned trace.
        </Typography.Text>

        {/* Existing questions */}
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
          {questions.map((q) => (
            <div
              key={q.name}
              css={{
                display: 'flex',
                alignItems: 'center',
                gap: theme.spacing.sm,
                padding: theme.spacing.sm,
                border: `1px solid ${theme.colors.border}`,
                borderRadius: theme.borders.borderRadiusMd,
              }}
            >
              <Input
                componentId={`${CID}.title`}
                value={q.title}
                onChange={(e) => retitle(q.name, e.target.value)}
                css={{ flex: 1 }}
              />
              <Tag componentId={`${CID}.type-tag`} color="charcoal">
                {TYPE_LABEL[q.type]}
                {q.options ? `: ${q.options.join(', ')}` : ''}
              </Tag>
              <Button
                componentId={`${CID}.remove`}
                icon={<TrashIcon />}
                aria-label={`Remove ${q.title}`}
                onClick={() => remove(q.name)}
              />
            </div>
          ))}
          {questions.length === 0 && <Typography.Hint>No questions yet. Add one below.</Typography.Hint>}
        </div>

        {/* Add a question */}
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.sm,
            borderTop: `1px solid ${theme.colors.border}`,
            paddingTop: theme.spacing.md,
          }}
        >
          <Typography.Text bold>Add a question</Typography.Text>
          <Input
            componentId={`${CID}.new-title`}
            value={newTitle}
            onChange={(e) => setNewTitle(e.target.value)}
            placeholder="Question title (e.g. Is the answer correct?)"
          />
          <div css={{ display: 'flex', gap: theme.spacing.xs }}>
            {TYPE_OPTIONS.map((opt) => (
              <Button
                key={opt.type}
                componentId={`${CID}.new-type`}
                size="small"
                type={newType === opt.type ? 'primary' : undefined}
                onClick={() => setNewType(opt.type)}
              >
                {opt.label}
              </Button>
            ))}
          </div>
          {newType === 'categorical' && (
            <Input
              componentId={`${CID}.new-options`}
              value={newOptions}
              onChange={(e) => setNewOptions(e.target.value)}
              placeholder="Comma-separated options (e.g. Low, Medium, High)"
            />
          )}
          <div>
            <Button componentId={`${CID}.add`} onClick={addQuestion} disabled={!canAdd}>
              Add question
            </Button>
          </div>
        </div>
      </div>
    </Modal>
  );
};
