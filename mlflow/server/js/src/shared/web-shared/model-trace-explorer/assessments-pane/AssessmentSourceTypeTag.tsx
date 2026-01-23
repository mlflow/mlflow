import { useDesignSystemTheme, Tag, SparkleIcon, CodeIcon, UserIcon } from '@databricks/design-system';

import type { AssessmentSourceType } from '../ModelTrace.types';
import { FormattedMessage } from 'react-intl';

export const AssessmentSourceTypeTag = ({ sourceType }: { sourceType?: AssessmentSourceType }) => {
  const { theme } = useDesignSystemTheme();

  if (sourceType === 'LLM_JUDGE') {
    return (
      <Tag
        componentId="shared.model-trace-explorer.llm-as-a-judge-tag"
        color="default"
        icon={<SparkleIcon />}
        css={{ width: 'fit-content', marginRight: 0 }}
      >
        <FormattedMessage defaultMessage="LLM-as-a-judge" description="Label for LLM scorer type" />
      </Tag>
    );
  }

  if (sourceType === 'CODE') {
    return (
      <Tag
        componentId="shared.model-trace-explorer.custom-code-scorer-tag"
        color="default"
        icon={<CodeIcon />}
        css={{ width: 'fit-content', marginRight: 0 }}
      >
        <FormattedMessage defaultMessage="Custom code scorer" description="Label for custom code scorer type" />
      </Tag>
    );
  }

  if (sourceType === 'HUMAN') {
    return (
      <Tag
        componentId="shared.model-trace-explorer.human-tag"
        color="default"
        icon={<UserIcon />}
        css={{ width: 'fit-content', marginRight: 0 }}
      >
        <FormattedMessage defaultMessage="Human feedback" description="Label for human feedback type" />
      </Tag>
    );
  }

  return null;
};
