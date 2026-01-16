/**
 * Floating Ask Assistant button component.
 * Positioned at bottom-right of the screen, always visible.
 */

import { useState } from 'react';
import { SparkleFillIcon, SparkleIcon, Tag, Tooltip, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { useAssistant } from './AssistantContext';

const COMPONENT_ID = 'mlflow.assistant.button';

export const AssistantButton = () => {
  const { theme } = useDesignSystemTheme();
  const { openPanel, isPanelOpen } = useAssistant();
  const [isHovered, setIsHovered] = useState(false);

  const handleClick = () => {
    openPanel();
  };

  return isPanelOpen ? null : (
    <Tooltip
      componentId={`${COMPONENT_ID}.tooltip`}
      content={
        <FormattedMessage
          defaultMessage="Ask Assistant for help"
          description="Tooltip for global Ask Assistant button"
        />
      }
    >
      <div
        data-assistant-ui="true"
        css={{
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.xs,
          cursor: 'pointer',
          borderRadius: theme.borders.borderRadiusMd,
          padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
          '&:hover': {
            backgroundColor: theme.colors.actionTertiaryBackgroundHover,
          },
        }}
        onClick={handleClick}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      >
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            transform: isHovered ? 'rotate(-90deg)' : 'rotate(0deg)',
            transition: 'transform 0.3s ease',
          }}
        >
          {isHovered ? (
            <SparkleFillIcon color="ai" css={{ fontSize: 20 }} />
          ) : (
            <SparkleIcon color="ai" css={{ fontSize: 20 }} />
          )}
        </div>
        <span css={{ fontWeight: theme.typography.typographyBoldFontWeight }}>
          <FormattedMessage defaultMessage="Assistant" description="Label for global Assistant button" />
        </span>
        <Tag componentId={`${COMPONENT_ID}.beta`} color="turquoise">
          Beta
        </Tag>
      </div>
    </Tooltip>
  );
};
