import type React from 'react';
import { useEffect, useMemo, useRef, useState } from 'react';

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

const isSubmitShortcut = (e: React.KeyboardEvent) => e.key === 'Enter' && (e.metaKey || e.ctrlKey);

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
        position: 'relative',
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.xs,
        padding: theme.spacing.sm,
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        backgroundColor: theme.colors.backgroundSecondary,
        '&::before': {
          content: '""',
          position: 'absolute',
          left: -theme.spacing.lg + theme.spacing.xs / 2,
          top: theme.spacing.md,
          width: theme.spacing.sm + 1,
          height: theme.spacing.sm + 1,
          borderRadius: '50%',
          backgroundColor: theme.colors.backgroundPrimary,
          border: `2px solid ${theme.colors.border}`,
          boxSizing: 'border-box',
        },
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
              content={<FormattedMessage defaultMessage="Edit comment" description="Tooltip for editing a comment" />}
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
                <FormattedMessage defaultMessage="Delete comment" description="Tooltip for deleting a comment" />
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
              defaultMessage: 'Edit comment... (⌘/Ctrl+Enter to save)',
              description: 'Placeholder text in the comments section edit input',
            })}
            onKeyDown={(e) => {
              e.stopPropagation();
              if (isSubmitShortcut(e) && editText.trim().length > 0 && !isUpdating) {
                e.preventDefault();
                saveEdit();
              }
            }}
            onChange={(e) => setEditText(e.target.value)}
          />
          <div css={{ display: 'flex', justifyContent: 'flex-end', gap: theme.spacing.xs }}>
            <Button
              componentId="shared.model-trace-explorer.assessment-note-edit-cancel"
              size="small"
              onClick={cancelEdit}
              disabled={isUpdating}
            >
              <FormattedMessage defaultMessage="Cancel" description="Button to cancel editing a comment" />
            </Button>
            <Button
              componentId="shared.model-trace-explorer.assessment-note-edit-save"
              size="small"
              type="primary"
              onClick={saveEdit}
              loading={isUpdating}
              disabled={editText.trim().length === 0 || isUpdating}
            >
              <FormattedMessage defaultMessage="Save" description="Button to save an edited comment" />
            </Button>
          </div>
        </>
      ) : (
        <CollapsibleNoteText text={text} />
      )}
    </div>
  );
};

const COLLAPSED_LINE_CLAMP = 5;

const CollapsibleNoteText = ({ text }: { text: string }) => {
  const { theme } = useDesignSystemTheme();
  const textRef = useRef<HTMLDivElement>(null);
  const [isExpanded, setIsExpanded] = useState(false);
  const [isOverflowing, setIsOverflowing] = useState(false);

  useEffect(() => {
    const el = textRef.current;
    if (!el) return;
    const measure = () => setIsOverflowing(el.scrollHeight > el.clientHeight + 1);
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, [text]);

  return (
    <div css={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: theme.spacing.xs }}>
      <div
        ref={textRef}
        css={{
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
          width: '100%',
          ...(!isExpanded && {
            display: '-webkit-box',
            WebkitLineClamp: COLLAPSED_LINE_CLAMP,
            WebkitBoxOrient: 'vertical',
            overflow: 'hidden',
          }),
        }}
      >
        <Typography.Text css={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>{text}</Typography.Text>
      </div>
      {isOverflowing && (
        <Typography.Link
          componentId="shared.model-trace-explorer.assessment-note-toggle-expand"
          onClick={() => setIsExpanded((v) => !v)}
        >
          {isExpanded ? (
            <FormattedMessage defaultMessage="Show less" description="Button to collapse a long comment" />
          ) : (
            <FormattedMessage defaultMessage="Show more" description="Button to expand a long comment" />
          )}
        </Typography.Link>
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
          <FormattedMessage
            defaultMessage="Comments"
            description="Header for the comments section in the assessments pane"
          />
        </Typography.Text>
        <Tooltip
          componentId="shared.model-trace-explorer.assessment-notes-info-tooltip"
          content={
            <FormattedMessage
              defaultMessage="Add comments about this trace. Each post is saved as a separate comment."
              description="Tooltip describing the comments section in the assessments pane"
            />
          }
        >
          <InfoSmallIcon css={{ color: theme.colors.textSecondary }} />
        </Tooltip>
      </div>
      <div
        css={{
          '& textarea.du-bois-light-input': {
            borderRadius: `${theme.borders.borderRadiusMd}px !important`,
          },
        }}
      >
        <Input.TextArea
          componentId="shared.model-trace-explorer.assessment-notes-input"
          value={draftText}
          autoSize={{ minRows: 3, maxRows: 10 }}
          disabled={isCreating}
          placeholder={intl.formatMessage({
            defaultMessage: 'Add a comment... (⌘/Ctrl+Enter to post)',
            description: 'Placeholder text in the comments section input',
          })}
          onKeyDown={(e) => {
            e.stopPropagation();
            if (isSubmitShortcut(e) && isDirty && !isCreating) {
              e.preventDefault();
              handlePost();
            }
          }}
          onChange={(e) => setDraftText(e.target.value)}
        />
      </div>
      <div css={{ display: 'flex', justifyContent: 'flex-end' }}>
        <Button
          componentId="shared.model-trace-explorer.assessment-notes-post"
          size="small"
          type="primary"
          onClick={handlePost}
          loading={isCreating}
          disabled={!isDirty || isCreating}
        >
          <FormattedMessage defaultMessage="Post" description="Button to post a new comment" />
        </Button>
      </div>
      {notes.length > 0 && (
        <div
          css={{
            position: 'relative',
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.sm,
            paddingLeft: theme.spacing.lg,
            '&::before': {
              content: '""',
              position: 'absolute',
              left: theme.spacing.sm - 1,
              top: theme.spacing.sm,
              bottom: theme.spacing.sm,
              width: 2,
              backgroundColor: theme.colors.border,
            },
          }}
        >
          {notes.map((note) => (
            <NoteItem key={note.assessment_id} note={note} canEdit={note.source.source_id === user} />
          ))}
        </div>
      )}
    </div>
  );
};
