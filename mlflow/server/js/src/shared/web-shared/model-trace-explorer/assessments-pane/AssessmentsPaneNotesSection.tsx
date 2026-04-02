import { useMemo, useState } from 'react';

import {
  Button,
  CheckIcon,
  InfoSmallIcon,
  Input,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import type { FeedbackAssessment } from '../ModelTrace.types';
import type { CreateAssessmentPayload, UpdateAssessmentPayload } from '../api';
import { getUser } from '../../global-settings/getUser';
import { useCreateAssessment } from '../hooks/useCreateAssessment';
import { useUpdateAssessment } from '../hooks/useUpdateAssessment';

export const NOTES_ASSESSMENT_NAME = 'mlflow.notes';

export const AssessmentsPaneNotesSection = ({
  traceId,
  feedbacks,
}: {
  traceId: string;
  feedbacks: FeedbackAssessment[];
}) => {
  const { theme } = useDesignSystemTheme();
  const user = getUser() ?? '';

  const existingNotes = useMemo(
    () => feedbacks.find((f) => f.assessment_name === NOTES_ASSESSMENT_NAME && f.source.source_id === user),
    [feedbacks, user],
  );

  const serverText = typeof existingNotes?.feedback?.value === 'string' ? existingNotes.feedback.value : '';

  const [notesText, setNotesText] = useState(serverText);
  const [prevServerText, setPrevServerText] = useState(serverText);

  // Sync from server when feedbacks prop updates (e.g. after save completes and
  // query refetch delivers updated data). This is React's recommended pattern for
  // adjusting state when props change without useEffect.
  // See: https://react.dev/learn/you-might-not-need-an-effect#adjusting-some-state-when-a-prop-changes
  if (serverText !== prevServerText) {
    setPrevServerText(serverText);
    setNotesText(serverText);
  }

  const { createAssessmentMutation, isLoading: isCreating } = useCreateAssessment({ traceId });
  const { updateAssessmentMutation, isLoading: isUpdating } = useUpdateAssessment({
    assessment: existingNotes as FeedbackAssessment,
  });

  const isLoading = isCreating || isUpdating;
  const isDirty = notesText !== serverText;

  const handleSave = () => {
    if (!isDirty) return;
    if (existingNotes) {
      const payload: UpdateAssessmentPayload = {
        assessment: { feedback: { value: notesText } },
        update_mask: 'feedback',
      };
      updateAssessmentMutation(payload);
    } else {
      const payload: CreateAssessmentPayload = {
        assessment: {
          assessment_name: NOTES_ASSESSMENT_NAME,
          trace_id: traceId,
          source: { source_type: 'HUMAN', source_id: user },
          feedback: { value: notesText },
        } as CreateAssessmentPayload['assessment'],
      };
      createAssessmentMutation(payload);
    }
  };

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs, marginTop: 'auto' }}>
      <div css={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
          <Typography.Text bold>
            <FormattedMessage
              defaultMessage="Notes"
              description="Header for the notes section in the assessments pane"
            />
          </Typography.Text>
          <Tooltip
            componentId="shared.model-trace-explorer.assessment-notes-info-tooltip"
            content={
              <FormattedMessage
                defaultMessage="Add personal notes about this trace."
                description="Tooltip describing the notes section in the assessments pane"
              />
            }
          >
            <InfoSmallIcon css={{ color: theme.colors.textSecondary }} />
          </Tooltip>
        </div>
        <Button
          componentId="shared.model-trace-explorer.assessment-notes-save"
          size="small"
          icon={<CheckIcon />}
          onClick={handleSave}
          loading={isLoading}
          disabled={!isDirty || isLoading}
        >
          <FormattedMessage defaultMessage="Save" description="Button to save notes assessment" />
        </Button>
      </div>
      <Input.TextArea
        componentId="shared.model-trace-explorer.assessment-notes-input"
        value={notesText}
        autoSize={{ minRows: 3, maxRows: 10 }}
        disabled={isLoading}
        onKeyDown={(e) => e.stopPropagation()}
        onChange={(e) => setNotesText(e.target.value)}
      />
    </div>
  );
};
