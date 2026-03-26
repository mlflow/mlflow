import { useEffect, useRef, useState } from 'react';

import { Button, ChevronDownIcon, ChevronRightIcon, useDesignSystemTheme } from '@databricks/design-system';

import { AssessmentDisplayValue } from './AssessmentDisplayValue';
import { FeedbackItem } from './FeedbackItem';
import { FeedbackValueGroupSourceCounts } from './FeedbackValueGroupSourceCounts';
import type { FeedbackAssessment } from '../ModelTrace.types';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';

export const FeedbackValueGroup = ({
  jsonValue,
  feedbacks,
}: {
  jsonValue: string;
  feedbacks: FeedbackAssessment[];
}) => {
  const { theme } = useDesignSystemTheme();
  const [expanded, setExpanded] = useState(false);
  const { subscribeToHighlightEvent } = useModelTraceExplorerViewState();
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const unsubscribes = feedbacks.map((f) =>
      subscribeToHighlightEvent(f.assessment_id, () => {
        setExpanded(true);
        containerRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }),
    );
    return () => unsubscribes.forEach((unsub) => unsub());
  }, [feedbacks, subscribeToHighlightEvent]);

  const assessmentName = feedbacks[0]?.assessment_name;

  return (
    <div ref={containerRef} css={{ display: 'flex', flexDirection: 'column' }}>
      <div css={{ display: 'flex', gap: theme.spacing.xs, alignItems: 'center' }}>
        <Button
          componentId="shared.model-trace-explorer.toggle-assessment-expanded"
          css={{ flexShrink: 0 }}
          size="small"
          icon={expanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
          onClick={() => setExpanded(!expanded)}
        />
        <AssessmentDisplayValue jsonValue={jsonValue} assessmentName={assessmentName} />
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
