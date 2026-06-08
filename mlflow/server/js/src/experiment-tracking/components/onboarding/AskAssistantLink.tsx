import { useState, type ReactNode } from 'react';
import { Typography, useDesignSystemTheme } from '@databricks/design-system';

import { AssistantSparkleIcon, useAssistant } from '../../../assistant';

export const AskAssistantLink = ({
  prompt,
  label,
  componentId,
}: {
  prompt: string;
  label: ReactNode;
  componentId: string;
}) => {
  const { theme } = useDesignSystemTheme();
  const { openPanel, reset, sendMessage, setupComplete } = useAssistant();
  const [isHovered, setIsHovered] = useState(false);

  const handleClick = (event: React.MouseEvent) => {
    event.preventDefault();
    openPanel();
    if (setupComplete) {
      reset();
      sendMessage(prompt);
    }
  };

  return (
    <Typography.Link
      componentId={componentId}
      onClick={handleClick}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      css={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: theme.spacing.xs,
        fontSize: 13,
        marginTop: theme.spacing.md,
      }}
    >
      <AssistantSparkleIcon isHovered={isHovered} iconSize={14} />
      {label}
    </Typography.Link>
  );
};
