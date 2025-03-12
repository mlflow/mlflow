import { ParagraphSkeleton, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { Link } from '../../../../common/utils/RoutingUtils';
import { usePromptVersionsForRunQuery } from '../../../pages/prompts/hooks/usePromptVersionsForRunQuery';
import Routes from '../../../routes';

export const RunViewRegisteredPromptsBox = ({ runUuid }: { runUuid: string }) => {
  const { theme } = useDesignSystemTheme();
  const { data, error, isLoading } = usePromptVersionsForRunQuery({ runUuid });
  const promptVersions = data?.model_versions;

  if (isLoading) {
    return <ParagraphSkeleton />;
  }

  if (error || !promptVersions || promptVersions.length === 0) {
    return <Typography.Hint>—</Typography.Hint>;
  }

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'row',
        gap: theme.spacing.sm,
        flexWrap: 'wrap',
        padding: `${theme.spacing.sm}px 0px`,
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
