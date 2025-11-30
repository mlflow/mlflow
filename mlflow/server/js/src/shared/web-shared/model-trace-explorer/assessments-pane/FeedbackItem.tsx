import { useState } from 'react';

import { useDesignSystemTheme } from '@databricks/design-system';

import { AssessmentEditForm } from './AssessmentEditForm';
import { AssessmentItemHeader } from './AssessmentItemHeader';
import { FeedbackItemContent } from './FeedbackItemContent';
import type { FeedbackAssessment } from '../ModelTrace.types';

export const FeedbackItem = ({ feedback }: { feedback: FeedbackAssessment }) => {
  const { theme } = useDesignSystemTheme();
  const [isEditing, setIsEditing] = useState(false);

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.xs,
        paddingLeft: theme.spacing.lg / 2,
        marginLeft: theme.spacing.lg / 2,
        paddingTop: theme.spacing.sm,
        paddingBottom: theme.spacing.sm,
        borderLeft: `1px solid ${theme.colors.border}`,
        position: 'relative',
      }}
    >
      <AssessmentItemHeader assessment={feedback} setIsEditing={setIsEditing} />
      {isEditing ? (
        <AssessmentEditForm
          assessment={feedback}
          onSuccess={() => setIsEditing(false)}
          onCancel={() => setIsEditing(false)}
        />
      ) : (
        <FeedbackItemContent feedback={feedback} />
      )}
    </div>
  );
};
