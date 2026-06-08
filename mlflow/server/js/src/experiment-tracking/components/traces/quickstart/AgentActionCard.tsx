import { useState, type ReactNode } from 'react';
import { Button, CopyIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

import { AssistantSparkleIcon, useAssistant } from '../../../../assistant';
import { CopyButton } from '@mlflow/mlflow/src/shared/building_blocks/CopyButton';

const TERMINAL_COMMAND = 'mlflow agent setup';

export const AgentActionCard = ({
  title,
  description,
  buttonLabel,
  prompt,
  componentId,
}: {
  title: ReactNode;
  description: ReactNode;
  buttonLabel: ReactNode;
  prompt: string;
  componentId: string;
}) => {
  const { theme } = useDesignSystemTheme();
  const { openPanel, reset, sendMessage, setupComplete } = useAssistant();
  const [isHovered, setIsHovered] = useState(false);

  const handleClick = () => {
    openPanel();
    if (setupComplete) {
      reset();
      sendMessage(prompt);
    }
  };

  return (
    <div
      css={{
        width: '100%',
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.md,
        padding: theme.spacing.lg,
        marginBottom: theme.spacing.lg * 1.5,
        borderRadius: theme.borders.borderRadiusMd,
        border: `1px solid ${theme.colors.border}`,
        backgroundColor: theme.colors.backgroundSecondary,
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.md }}>
        <AssistantSparkleIcon isHovered={isHovered} iconSize={24} />
        <div css={{ flex: 1, minWidth: 0 }}>
          <Typography.Title level={4} css={{ marginBottom: theme.spacing.xs }}>
            {title}
          </Typography.Title>
          <Typography.Text color="secondary" css={{ fontSize: 13 }}>
            {description}
          </Typography.Text>
        </div>
        <Button componentId={componentId} type="primary" onClick={handleClick}>
          {buttonLabel}
        </Button>
      </div>

      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.sm,
          paddingLeft: theme.spacing.lg * 1.5,
        }}
      >
        <Typography.Text color="secondary" css={{ fontSize: 12 }}>
          <FormattedMessage
            defaultMessage="Or run in your terminal:"
            description="Label preceding the mlflow agent setup terminal command on the agent action card"
          />
        </Typography.Text>
        <code
          css={{
            fontSize: 12,
            padding: `2px ${theme.spacing.sm}px`,
            borderRadius: theme.borders.borderRadiusSm,
            backgroundColor: theme.colors.backgroundPrimary,
            border: `1px solid ${theme.colors.border}`,
            color: theme.colors.textPrimary,
          }}
        >
          {TERMINAL_COMMAND}
        </code>
        <CopyButton
          componentId={`${componentId}.copy_agent_setup_command`}
          copyText={TERMINAL_COMMAND}
          showLabel={false}
          icon={<CopyIcon />}
        />
      </div>
    </div>
  );
};
