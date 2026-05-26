import { useMemo, useState } from 'react';

import {
  Button,
  InfoSmallIcon,
  Input,
  PencilIcon,
  Tooltip,
  Typography,
  TrashIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';

import type { FeedbackAssessment } from '../ModelTrace.types';
import type { CreateAssessmentPayload, UpdateAssessmentPayload } from '../api';
import { getUser } from '../../global-settings/getUser';
import { useCreateAssessment } from '../hooks/useCreateAssessment';
import { useDeleteAssessment } from '../hooks/useDeleteAssessment';
import { useUpdateAssessment } from '../hooks/useUpdateAssessment';
import { timeSinceStr } from './AssessmentsPane.utils';

export const NOTES_ASSESSMENT_NAME = 'mlflow.notes';

const NoteItem = ({ note, canEdit }: { note: FeedbackAssessment; canEdit: boolean }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const text = typeof note.feedback?.value === 'string' ? note.feedback.value : '';

  const [isEditing, setIsEditing] = useState(false);
  const [editText, setEditText] = useState(text);

  const { deleteAssessmentMutation, isLoading: isDeleting } = useDeleteAssessment({ assessment: note });
  const { updateAssessmentMutation, isLoading: isUpdating } = useUpdateAssessment({
    assessment: note,
    onSuccess: () => setIsEditing(false),
  });

  const startEdit = () => {
    setEditText(text);
    setIsEditing(true);
  };

  const cancelEdit = () => {
    setEditText(text);
    setIsEditing(false);
  };

  const saveEdit = () => {
    if (editText === text) {
      setIsEditing(false);
      return;
    }
    const payload: UpdateAssessmentPayload = {
      assessment: { feedback: { value: editText } },
      update_mask: 'feedback',
    };
    updateAssessmentMutation(payload);
  };

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.xs,
        padding: theme.spacing.sm,
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        backgroundColor: theme.colors.backgroundSecondary,
      }}
    >
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <Typography.Text bold size="sm">
          {note.source.source_id}
        </Typography.Text>
        {note.last_update_time && (
          <Typography.Text color="secondary" size="sm">
            {timeSinceStr(new Date(note.last_update_time))}
          </Typography.Text>
        )}
        {canEdit && !isEditing && (
          <div css={{ marginLeft: 'auto', display: 'flex', gap: theme.spacing.xs }}>
            <Tooltip
              componentId="shared.model-trace-explorer.assessment-note-edit-tooltip"
              content={
                <FormattedMessage defaultMessage="Edit note" description="Tooltip for editing a notes comment" />
              }
            >
              <Button
                componentId="shared.model-trace-explorer.assessment-note-edit"
                size="small"
                icon={<PencilIcon />}
                disabled={isDeleting}
                onClick={startEdit}
              />
            </Tooltip>
            <Tooltip
              componentId="shared.model-trace-explorer.assessment-note-delete-tooltip"
              content={
                <FormattedMessage defaultMessage="Delete note" description="Tooltip for deleting a notes comment" />
              }
            >
              <Button
                componentId="shared.model-trace-explorer.assessment-note-delete"
                size="small"
                icon={<TrashIcon />}
                loading={isDeleting}
                onClick={() => deleteAssessmentMutation()}
              />
            </Tooltip>
          </div>
        )}
      </div>
      {isEditing ? (
        <>
          <Input.TextArea
            componentId="shared.model-trace-explorer.assessment-note-edit-input"
            value={editText}
            autoSize={{ minRows: 2, maxRows: 10 }}
            disabled={isUpdating}
            placeholder={intl.formatMessage({
              defaultMessage: 'Edit note...',
              description: 'Placeholder text in the notes section edit input',
            })}
            onKeyDown={(e) => e.stopPropagation()}
            onChange={(e) => setEditText(e.target.value)}
          />
          <div css={{ display: 'flex', justifyContent: 'flex-end', gap: theme.spacing.xs }}>
            <Button
              componentId="shared.model-trace-explorer.assessment-note-edit-cancel"
              size="small"
              onClick={cancelEdit}
              disabled={isUpdating}
            >
              <FormattedMessage defaultMessage="Cancel" description="Button to cancel editing a notes comment" />
            </Button>
            <Button
              componentId="shared.model-trace-explorer.assessment-note-edit-save"
              size="small"
              type="primary"
              onClick={saveEdit}
              loading={isUpdating}
              disabled={editText.trim().length === 0 || isUpdating}
            >
              <FormattedMessage defaultMessage="Save" description="Button to save an edited notes comment" />
            </Button>
          </div>
        </>
      ) : (
        <Typography.Text css={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>{text}</Typography.Text>
      )}
    </div>
  );
};

export const AssessmentsPaneNotesSection = ({
  traceId,
  feedbacks,
}: {
  traceId: string;
  feedbacks: FeedbackAssessment[];
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const user = getUser() ?? '';

  const notes = useMemo(
    () =>
      feedbacks
        .filter((f) => f.assessment_name === NOTES_ASSESSMENT_NAME)
        .slice()
        .sort((a, b) => new Date(b.create_time).getTime() - new Date(a.create_time).getTime()),
    [feedbacks],
  );

  const [draftText, setDraftText] = useState('');

  const { createAssessmentMutation, isLoading: isCreating } = useCreateAssessment({
    traceId,
    onSuccess: () => setDraftText(''),
  });

  const isDirty = draftText.trim().length > 0;

  const handlePost = () => {
    if (!isDirty) return;
    const payload: CreateAssessmentPayload = {
      assessment: {
        assessment_name: NOTES_ASSESSMENT_NAME,
        trace_id: traceId,
        source: { source_type: 'HUMAN', source_id: user },
        feedback: { value: draftText },
      } as CreateAssessmentPayload['assessment'],
    };
    createAssessmentMutation(payload);
  };

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm, marginTop: 'auto' }}>
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
        <Typography.Text bold>
          <FormattedMessage defaultMessage="Notes" description="Header for the notes section in the assessments pane" />
        </Typography.Text>
        <Tooltip
          componentId="shared.model-trace-explorer.assessment-notes-info-tooltip"
          content={
            <FormattedMessage
              defaultMessage="Add notes about this trace. Each post is saved as a separate comment."
              description="Tooltip describing the notes section in the assessments pane"
            />
          }
        >
          <InfoSmallIcon css={{ color: theme.colors.textSecondary }} />
        </Tooltip>
      </div>
      <Input.TextArea
        componentId="shared.model-trace-explorer.assessment-notes-input"
        value={draftText}
        autoSize={{ minRows: 3, maxRows: 10 }}
        disabled={isCreating}
        placeholder={intl.formatMessage({
          defaultMessage: 'Add a note...',
          description: 'Placeholder text in the notes section input',
        })}
        onKeyDown={(e) => e.stopPropagation()}
        onChange={(e) => setDraftText(e.target.value)}
      />
      <div css={{ display: 'flex', justifyContent: 'flex-end' }}>
        <Button
          componentId="shared.model-trace-explorer.assessment-notes-post"
          size="small"
          type="primary"
          onClick={handlePost}
          loading={isCreating}
          disabled={!isDirty || isCreating}
        >
          <FormattedMessage defaultMessage="Post" description="Button to post a new notes comment" />
        </Button>
      </div>
      {notes.length > 0 && (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          {notes.map((note) => (
            <NoteItem key={note.assessment_id} note={note} canEdit={note.source.source_id === user} />
          ))}
        </div>
      )}
    </div>
  );
};
