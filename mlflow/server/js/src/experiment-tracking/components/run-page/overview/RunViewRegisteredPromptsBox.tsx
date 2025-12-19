import { ParagraphSkeleton, Typography, useDesignSystemTheme } from '@databricks/design-system';
import type { KeyValueEntity } from '@mlflow/mlflow/src/common/types';

import { Link } from '../../../../common/utils/RoutingUtils';
import { usePromptVersionsForRunQuery } from '../../../pages/prompts/hooks/usePromptVersionsForRunQuery';
import Routes from '../../../routes';
import { parseLinkedPromptsFromRunTags } from '../../../pages/prompts/utils';

export const RunViewRegisteredPromptsBox = ({
  tags,
  runUuid,
}: {
  tags: Record<string, KeyValueEntity>;
  runUuid: string;
}) => {
  const { theme } = useDesignSystemTheme();
  // This part is for supporting prompt versions created using mlflow < 3.1.0
  const { data, error, isLoading } = usePromptVersionsForRunQuery({ runUuid });
  const promptVersionsFromPromptTags = data?.model_versions || [];
  const promptVersionsFromRunTags = parseLinkedPromptsFromRunTags(tags);
  const promptVersions = [...promptVersionsFromPromptTags, ...promptVersionsFromRunTags];

  if (isLoading) {
    return <ParagraphSkeleton />;
  }

  if (error || !promptVersions || promptVersions.length === 0) {
    return <Typography.Hint css={{ padding: `${theme.spacing.xs}px 0px` }}>—</Typography.Hint>;
  }

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'row',
        gap: theme.spacing.sm,
        flexWrap: 'wrap',
        padding: `${theme.spacing.xs}px 0px`,
      }}
    >
      {promptVersions.map((promptVersion, index) => {
        const to = Routes.getPromptDetailsPageRoute(encodeURIComponent(promptVersion.name));
        const displayText = `${promptVersion.name} (v${promptVersion.version})`;
        return (
          <Typography.Text key={displayText} css={{ whiteSpace: 'nowrap' }}>
            <Link to={to}>{displayText}</Link>
            {index < promptVersions.length - 1 && ','}
          </Typography.Text>
        );
      })}
    </div>
  );
};
