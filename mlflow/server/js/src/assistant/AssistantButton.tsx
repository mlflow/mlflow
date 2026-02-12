/**
 * Floating Ask Assistant button component.
 * Positioned at bottom-right of the screen, always visible.
 */

import { useState } from 'react';
import {
  Button,
  CloseIcon,
  SparkleFillIcon,
  SparkleIcon,
  Tag,
  Tooltip,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { useAssistant } from './AssistantContext';

const COMPONENT_ID = 'mlflow.assistant.button';

export const AssistantButton = () => {
  const { theme } = useDesignSystemTheme();
  const {
    gradientStart: aiGradientStart,
    gradientMid: aiGradientMid,
    gradientEnd: aiGradientEnd,
  } = theme.colors.branded.ai;
  const { openPanel, isPanelOpen, isButtonDismissed, dismissButton } = useAssistant();
  const [isHovered, setIsHovered] = useState(false);

  const handleClick = () => {
    openPanel();
  };

  const handleDismiss = (e: React.MouseEvent) => {
    e.stopPropagation();
    dismissButton();
  };

  if (isPanelOpen || isButtonDismissed) {
    return null;
  }

  return (
    <div
      data-assistant-ui="true"
      css={{
        position: 'fixed',
        bottom: theme.spacing.lg,
        right: theme.spacing.lg,
        // NB: Must be higher than the Drawer's z-index
        zIndex: theme.options.zIndexBase + 100,
      }}
    >
      <Tooltip
        componentId={`${COMPONENT_ID}.tooltip`}
        content={
          <FormattedMessage
            defaultMessage="Ask Assistant for help"
            description="Tooltip for global Ask Assistant button"
          />
        }
      >
        {/* Rainbow gradient border wrapper */}
        <div
          css={{
            background: `linear-gradient(90deg, ${aiGradientStart}, ${aiGradientMid}, ${aiGradientEnd})`,
            borderRadius: 24,
            padding: 2, // Border width
            boxShadow: '0 4px 16px rgba(0, 0, 0, 0.15)',
            cursor: 'pointer',
            transition: 'box-shadow 0.2s ease, transform 0.2s ease',
            '&:hover': {
              boxShadow: '0 6px 24px rgba(0, 0, 0, 0.2)',
              transform: 'translateY(-2px)',
            },
          }}
          onClick={handleClick}
          onMouseEnter={() => setIsHovered(true)}
          onMouseLeave={() => setIsHovered(false)}
        >
          <div css={{ display: 'flex', alignItems: 'center', gap: 0 }}>
            <Button
              componentId={COMPONENT_ID}
              icon={
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
              }
              css={{
                backgroundColor: '#ffffff !important',
                border: 'none !important',
                borderRadius: '22px 0 0 22px',
                height: 'auto',
                minHeight: 48,
                fontSize: theme.typography.fontSizeMd,
                display: 'flex',
                alignItems: 'center',
                gap: theme.spacing.xs,
                paddingRight: theme.spacing.sm,
                '&:hover': {
                  backgroundColor: '#fafafa !important',
                },
              }}
            >
              <FormattedMessage defaultMessage="Assistant" description="Label for global Assistant button" />
              <Tag componentId={`${COMPONENT_ID}.beta`} color="turquoise" css={{ marginLeft: 4 }}>
                Beta
              </Tag>
            </Button>
            <Tooltip
              componentId={`${COMPONENT_ID}.dismiss.tooltip`}
              content={
                <FormattedMessage defaultMessage="Dismiss" description="Tooltip for dismissing the Assistant button" />
              }
            >
              <button
                onClick={handleDismiss}
                css={{
                  backgroundColor: '#ffffff',
                  border: 'none',
                  borderLeft: `1px solid ${theme.colors.border}`,
                  borderRadius: '0 22px 22px 0',
                  height: 48,
                  width: 36,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  cursor: 'pointer',
                  color: theme.colors.textSecondary,
                  transition: 'background-color 0.2s ease, color 0.2s ease',
                  '&:hover': {
                    backgroundColor: '#fafafa',
                    color: theme.colors.textPrimary,
                  },
                }}
              >
                <CloseIcon css={{ fontSize: 14 }} />
              </button>
            </Tooltip>
          </div>
        </div>
      </Tooltip>
    </div>
  );
};
