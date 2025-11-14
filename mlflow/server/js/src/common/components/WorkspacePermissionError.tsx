import { useDesignSystemTheme } from '@databricks/design-system';
import { ErrorView } from './ErrorView';
import { WorkspaceSelector } from './WorkspaceSelector';

export const WorkspacePermissionError = ({ workspaceName }: { workspaceName: string }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: theme.spacing.lg,
        padding: theme.spacing.lg,
      }}
    >
      <ErrorView
        statusCode={403}
        subMessage={`You don't have access to workspace: ${workspaceName}. Please select another workspace.`}
      />
      <WorkspaceSelector />
    </div>
  );
};
