import React from 'react';
import { Button, CloseIcon, Popover, SparkleIcon, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useLocalStorage } from '../../hooks/useLocalStorage';

// Default storage key for tracking first-time user guidance
export const DEFAULT_DETECT_ISSUES_GUIDANCE_STORAGE_KEY = 'mlflow.detectIssues.guidanceShown';
const DETECT_ISSUES_GUIDANCE_STORAGE_VERSION = 1;

interface DetectIssuesButtonProps {
  componentId: string;
  onClick: () => void;
}

export const DetectIssuesButton: React.FC<DetectIssuesButtonProps> = ({ componentId, onClick }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  // Track whether the user has seen the guidance
  const [hasSeenGuidance, setHasSeenGuidance] = useLocalStorage({
    key: DEFAULT_DETECT_ISSUES_GUIDANCE_STORAGE_KEY,
    version: DETECT_ISSUES_GUIDANCE_STORAGE_VERSION,
    initialValue: false,
  });

  const handleDismissGuidance = () => {
    setHasSeenGuidance(true);
  };

  const buttonContent = (
    <Button
      componentId={componentId}
      onClick={onClick}
      aria-label={intl.formatMessage({
        defaultMessage: 'Detect issues in traces',
        description: 'Aria label for the detect issues button',
      })}
      icon={<SparkleIcon color="ai" />}
      css={{
        border: '1px solid transparent !important',
        background: `linear-gradient(${theme.colors.backgroundPrimary}, ${theme.colors.backgroundPrimary}) padding-box, ${theme.gradients.aiBorderGradient} border-box`,
      }}
    >
      <FormattedMessage defaultMessage="Detect Issues" description="Label for the detect issues button" />
    </Button>
  );

  // Wrap in Popover if guidance should be shown
  if (!hasSeenGuidance) {
    return (
      <Popover.Root
        componentId="mlflow.detect_issues.guidance"
        open={!hasSeenGuidance}
        onOpenChange={(open) => {
          // Only allow closing via explicit dismiss actions, not by clicking outside
          if (!open) {
            return;
          }
        }}
        modal
      >
        <Popover.Trigger asChild>
          <div css={{ position: 'relative', zIndex: 1001 }}>{buttonContent}</div>
        </Popover.Trigger>
        <Popover.Content
          side="bottom"
          align="end"
          onEscapeKeyDown={(e) => e.preventDefault()}
          onPointerDownOutside={(e) => e.preventDefault()}
          onInteractOutside={(e) => e.preventDefault()}
          css={{
            maxWidth: 320,
            padding: theme.spacing.md,
            zIndex: 1000,
            boxShadow: '0 0 0 9999px rgba(0, 0, 0, 0.5), 0 8px 24px rgba(0, 0, 0, 0.4)',
          }}
        >
          <Popover.Arrow css={{ fill: theme.colors.backgroundPrimary }} />
          <div
            css={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'flex-start',
              marginBottom: theme.spacing.sm,
            }}
          >
            <div
              css={{
                fontWeight: theme.typography.typographyBoldFontWeight,
                fontSize: theme.typography.fontSizeMd,
              }}
            >
              <FormattedMessage
                defaultMessage="Detect Issues in Your Traces"
                description="Detect issues guidance popover title"
              />
            </div>
            <Button
              componentId="mlflow.detect_issues.guidance.dismiss"
              icon={<CloseIcon />}
              onClick={handleDismissGuidance}
              aria-label={intl.formatMessage({
                defaultMessage: 'Close guidance',
                description: 'Aria label for closing the detect issues guidance popover',
              })}
              css={{
                padding: 0,
                minWidth: 'auto',
                border: 'none',
                background: 'transparent',
              }}
            />
          </div>
          <div css={{ fontSize: theme.typography.fontSizeSm, color: theme.colors.textSecondary }}>
            <FormattedMessage
              defaultMessage="Use this button to automatically detect quality issues in your traces using AI. This feature helps you identify problems like correctness, latency, and other quality concerns in your agent."
              description="Detect issues guidance popover message"
            />
          </div>
          <Button
            componentId="mlflow.detect_issues.guidance.got_it"
            onClick={handleDismissGuidance}
            css={{ marginTop: theme.spacing.md, width: '100%' }}
          >
            <FormattedMessage defaultMessage="Got it" description="Detect issues guidance dismiss button" />
          </Button>
        </Popover.Content>
      </Popover.Root>
    );
  }

  return <div>{buttonContent}</div>;
};
