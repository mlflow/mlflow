/**
 * Floating Ask Assistant button component.
 * Positioned at bottom-right of the screen, always visible.
 */

import { useState } from 'react';
import { Button, SparkleFillIcon, SparkleIcon, Tag, Tooltip, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { useAssistant } from './AssistantContext';

const COMPONENT_ID = 'mlflow.assistant.button';
const RAINBOW_GRADIENT = 'linear-gradient(90deg, #64B5F6, #BA68C8, #E57373)';

export const AssistantButton = () => {
  const { theme } = useDesignSystemTheme();
  const { openPanel, isPanelOpen } = useAssistant();
  const [isHovered, setIsHovered] = useState(false);

  const handleClick = () => {
    openPanel();
  };

  return isPanelOpen ? null : (
    <div
      data-assistant-ui="true"
      css={{
        position: 'fixed',
        bottom: theme.spacing.lg,
        right: theme.spacing.lg + theme.spacing.sm,
        zIndex: 1500,
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
            background: RAINBOW_GRADIENT,
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
              borderRadius: 24 - 2,
              height: 'auto',
              minHeight: 48,
              fontSize: theme.typography.fontSizeMd,
              display: 'flex',
              alignItems: 'center',
              gap: theme.spacing.xs,
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
        </div>
      </Tooltip>
    </div>
  );
};
