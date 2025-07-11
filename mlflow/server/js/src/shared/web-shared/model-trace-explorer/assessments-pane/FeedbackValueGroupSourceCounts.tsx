import { countBy } from 'lodash';

import { CodeIcon, SparkleIcon, Tag, Typography, useDesignSystemTheme, UserIcon } from '@databricks/design-system';

import type { AssessmentSourceType, FeedbackAssessment } from '../ModelTrace.types';

const getSourceTypeIcon = (sourceType: AssessmentSourceType) => {
  const smallIconStyles = {
    '& > svg': {
      width: 12,
      height: 12,
    },
  };
  switch (sourceType) {
    case 'HUMAN':
      return <UserIcon css={smallIconStyles} />;
    case 'LLM_JUDGE':
      return <SparkleIcon css={smallIconStyles} />;
    case 'CODE':
      return <CodeIcon css={smallIconStyles} />;
    default:
      return null;
  }
};

export const FeedbackValueGroupSourceCounts = ({ feedbacks }: { feedbacks: FeedbackAssessment[] }) => {
  const { theme } = useDesignSystemTheme();

  if (feedbacks.length < 2) {
    return null;
  }

  const sourceCounts = countBy(feedbacks, (feedback) => feedback.source.source_type);
  return (
    <div css={{ display: 'flex', gap: theme.spacing.xs, alignItems: 'center', marginLeft: theme.spacing.xs }}>
      {Object.entries(sourceCounts).map(([sourceType, count]) => (
        <Tag
          componentId={`shared.model-trace-explorer.feedback-source-count-${sourceType}`}
          css={{
            margin: 0,
          }}
          key={sourceType}
        >
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
            {getSourceTypeIcon(sourceType as AssessmentSourceType)}
            <Typography.Text>{count}</Typography.Text>
          </div>
        </Tag>
      ))}
    </div>
  );
};
