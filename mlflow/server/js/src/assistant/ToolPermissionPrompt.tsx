/**
 * Inline Yes/No prompt shown when the assistant wants to run a tool while the
 * session is not in full-access mode. The backend stream stays paused until the
 * user answers.
 */

import { Button, Typography, WrenchSparkleIcon, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import type { PermissionRequest } from './types';

/**
 * Pick the most meaningful single line of the tool input to show the user:
 * the command for Bash, the file path for file tools, else the raw input.
 */
const getInputPreview = (toolName: string, toolInput: Record<string, any>): string => {
  if (toolName === 'Bash' && typeof toolInput['command'] === 'string') {
    return toolInput['command'];
  }
  const filePath = toolInput['file_path'] ?? toolInput['path'];
  if (typeof filePath === 'string') {
    return filePath;
  }
  try {
    return JSON.stringify(toolInput);
  } catch {
    return '';
  }
};

export const ToolPermissionPrompt = ({
  request,
  onRespond,
}: {
  request: PermissionRequest;
  onRespond: (allow: boolean) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const preview = getInputPreview(request.toolName, request.toolInput);

  return (
    <div
      role="group"
      aria-label="Tool permission request"
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.sm,
        marginBottom: theme.spacing.sm,
        padding: theme.spacing.md,
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        backgroundColor: theme.colors.backgroundSecondary,
      }}
    >
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <WrenchSparkleIcon color="ai" css={{ fontSize: 16, flexShrink: 0 }} />
        <Typography.Text bold>
          <FormattedMessage
            defaultMessage="Allow the assistant to run {toolName}?"
            description="Title of the prompt asking the user to approve a tool call"
            values={{ toolName: request.toolName }}
          />
        </Typography.Text>
      </div>

      {preview && (
        <Typography.Text
          code
          css={{
            display: 'block',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
            maxHeight: 120,
            overflowY: 'auto',
            color: theme.colors.textSecondary,
          }}
        >
          {preview}
        </Typography.Text>
      )}

      <div css={{ display: 'flex', justifyContent: 'flex-end', gap: theme.spacing.sm }}>
        <Button componentId="mlflow.assistant.permission.deny" onClick={() => onRespond(false)}>
          <FormattedMessage defaultMessage="Deny" description="Button to deny a tool-call permission request" />
        </Button>
        <Button componentId="mlflow.assistant.permission.allow" type="primary" onClick={() => onRespond(true)}>
          <FormattedMessage defaultMessage="Allow" description="Button to allow a tool-call permission request" />
        </Button>
      </div>
    </div>
  );
};
