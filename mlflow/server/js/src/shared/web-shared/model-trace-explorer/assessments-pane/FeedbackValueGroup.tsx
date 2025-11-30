import { useState } from 'react';

import { Button, ChevronDownIcon, ChevronRightIcon, useDesignSystemTheme } from '@databricks/design-system';

import { AssessmentDisplayValue } from './AssessmentDisplayValue';
import { FeedbackItem } from './FeedbackItem';
import { FeedbackValueGroupSourceCounts } from './FeedbackValueGroupSourceCounts';
import type { FeedbackAssessment } from '../ModelTrace.types';

export const FeedbackValueGroup = ({
  jsonValue,
  feedbacks,
}: {
  jsonValue: string;
  feedbacks: FeedbackAssessment[];
}) => {
  const { theme } = useDesignSystemTheme();
  const [expanded, setExpanded] = useState(false);

  return (
    <div css={{ display: 'flex', flexDirection: 'column' }}>
      <div css={{ display: 'flex', gap: theme.spacing.xs, alignItems: 'center' }}>
        <Button
          componentId="shared.model-trace-explorer.toggle-assessment-expanded"
          css={{ flexShrink: 0 }}
          size="small"
          icon={expanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
          onClick={() => setExpanded(!expanded)}
        />
        <AssessmentDisplayValue jsonValue={jsonValue} />
        <FeedbackValueGroupSourceCounts feedbacks={feedbacks} />
      </div>
      {expanded && (
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          {feedbacks.map((feedback) =>
            // don't display assessments that have been overridden
            feedback?.valid === false ? null : <FeedbackItem feedback={feedback} key={feedback.assessment_id} />,
          )}
        </div>
      )}
    </div>
  );
};
