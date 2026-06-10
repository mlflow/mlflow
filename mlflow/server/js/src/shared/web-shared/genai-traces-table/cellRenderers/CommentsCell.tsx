import { SpeechBubbleIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';

import { NullCell } from './NullCell';
import { StackedComponents } from './StackedComponents';
import type { RunEvaluationResultAssessment } from '../types';
import { NOTES_ASSESSMENT_NAME } from '../../model-trace-explorer/assessments-pane/AssessmentsPaneNotesSection';

const CommentsCount = ({
  assessments,
  isComparing,
}: {
  assessments: RunEvaluationResultAssessment[] | undefined;
  isComparing: boolean;
}) => {
  const { theme } = useDesignSystemTheme();
  const count = assessments?.length ?? 0;

  if (count === 0) {
    return <NullCell isComparing={isComparing} />;
  }

  const latestText =
    count === 1 && typeof assessments?.[0]?.stringValue === 'string' ? assessments[0].stringValue : undefined;

  return (
    <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs, minWidth: 0 }}>
      <SpeechBubbleIcon css={{ color: theme.colors.textSecondary, flexShrink: 0 }} />
      {latestText ? (
        <Typography.Text
          css={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', minWidth: 0 }}
          title={latestText}
        >
          {latestText}
        </Typography.Text>
      ) : (
        <Typography.Text css={{ flexShrink: 0 }}>{count}</Typography.Text>
      )}
    </div>
  );
};

export const CommentsCell = ({
  currentResponseAssessmentsByName,
  otherResponseAssessmentsByName,
  isComparing,
}: {
  currentResponseAssessmentsByName: Record<string, RunEvaluationResultAssessment[]> | undefined;
  otherResponseAssessmentsByName: Record<string, RunEvaluationResultAssessment[]> | undefined;
  isComparing: boolean;
}) => {
  return (
    <StackedComponents
      first={
        <CommentsCount
          assessments={currentResponseAssessmentsByName?.[NOTES_ASSESSMENT_NAME]}
          isComparing={isComparing}
        />
      }
      second={
        isComparing && (
          <CommentsCount
            assessments={otherResponseAssessmentsByName?.[NOTES_ASSESSMENT_NAME]}
            isComparing={isComparing}
          />
        )
      }
    />
  );
};
