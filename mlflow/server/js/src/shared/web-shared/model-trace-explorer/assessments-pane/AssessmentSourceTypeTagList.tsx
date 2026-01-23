import { useDesignSystemTheme } from '@databricks/design-system';

import type { FeedbackAssessment } from '../ModelTrace.types';
import { uniq } from 'lodash';
import { AssessmentSourceTypeTag } from './AssessmentSourceTypeTag';

export const AssessmentSourceTypeTagList = ({ assessments }: { assessments: FeedbackAssessment[] }) => {
  const { theme } = useDesignSystemTheme();
  const assessmentSourceTypes = uniq(assessments.map((assessment) => assessment.source.source_type));

  return (
    <div css={{ display: 'flex', gap: theme.spacing.xs, alignItems: 'center' }}>
      {assessmentSourceTypes.map((sourceType) => (
        <AssessmentSourceTypeTag key={sourceType} sourceType={sourceType} />
      ))}
    </div>
  );
};
