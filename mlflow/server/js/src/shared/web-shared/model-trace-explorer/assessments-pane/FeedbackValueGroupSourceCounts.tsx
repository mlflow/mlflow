import { countBy } from 'lodash';

import {
  CodeIcon,
  SparkleIcon,
  Tag,
  Tooltip,
  Typography,
  useDesignSystemTheme,
  UserIcon,
} from '@databricks/design-system';

import type { AssessmentSourceType, FeedbackAssessment } from '../ModelTrace.types';
import { FormattedMessage } from 'react-intl';

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

const getSourceTypeTooltipContent = (sourceType: AssessmentSourceType) => {
  switch (sourceType) {
    case 'HUMAN':
      return <FormattedMessage defaultMessage="Human feedback" description="Tooltip content for human feedback" />;
    case 'LLM_JUDGE':
      return (
        <FormattedMessage defaultMessage="LLM judge feedback" description="Tooltip content for LLM judge feedback" />
      );
    case 'CODE':
      return (
        <FormattedMessage
          defaultMessage="Custom code judge feedback"
          description="Tooltip content for custom code judge feedback"
        />
      );
  }
  return null;
};

export const FeedbackValueGroupSourceCounts = ({ feedbacks }: { feedbacks: FeedbackAssessment[] }) => {
  const { theme } = useDesignSystemTheme();

  if (feedbacks.length < 1) {
    return null;
  }

  const sourceCounts = countBy(feedbacks, (feedback) => feedback.source.source_type);
  return (
    <div
      css={{
        display: 'flex',
        gap: theme.spacing.xs,
        alignItems: 'center',
        marginLeft: theme.spacing.xs,
      }}
    >
      {Object.entries(sourceCounts).map(([sourceType, count]) => (
        <Tooltip
          componentId="shared.model-trace-explorer.feedback-source-tooltip"
          content={getSourceTypeTooltipContent(sourceType as AssessmentSourceType)}
        >
          <Tag
            componentId="shared.model-trace-explorer.feedback-source-count"
            css={{
              margin: 0,
              '&>*': {
                cursor: 'default',
              },
            }}
            key={sourceType}
          >
            <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
              {getSourceTypeIcon(sourceType as AssessmentSourceType)}
              {count > 1 && <Typography.Text>{count}</Typography.Text>}
            </div>
          </Tag>
        </Tooltip>
      ))}
    </div>
  );
};
