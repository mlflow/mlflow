import React, { useEffect, useState } from 'react';
import { Button, CloseIcon, Popover, SparkleIcon, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { getLocalStorageItemByParams, useLocalStorage } from '../../hooks/useLocalStorage';
import { shouldEnableIssueDetection } from '../../../../common/utils/FeatureUtils';
import {
  DEFAULT_DETECT_ISSUES_GUIDANCE_STORAGE_KEY,
  DETECT_ISSUES_GUIDANCE_STORAGE_VERSION,
  DETECT_ISSUES_GUIDANCE_DISMISSED_EVENT,
} from './DetectIssuesButton';

const ANALYZE_WITH_ASSISTANT_GUIDANCE_STORAGE_KEY = 'mlflow.assistant.tracesGuidanceShown';
const ANALYZE_WITH_ASSISTANT_GUIDANCE_STORAGE_VERSION = 1;

interface AnalyzeWithAssistantButtonProps {
  componentId: string;
  onClick: () => void;
}

export const AnalyzeWithAssistantButton: React.FC<AnalyzeWithAssistantButtonProps> = ({ componentId, onClick }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const [hasSeenGuidance, setHasSeenGuidance] = useLocalStorage({
    key: ANALYZE_WITH_ASSISTANT_GUIDANCE_STORAGE_KEY,
    version: ANALYZE_WITH_ASSISTANT_GUIDANCE_STORAGE_VERSION,
    initialValue: false,
  });

  // Wait for the Detect Issues spotlight to be dismissed before showing ours, so the two
  // first-visit callouts appear one at a time rather than stacking dimming scrims and focus
  // traps. When issue detection is disabled, Detect Issues never shows, so we go immediately.
  // useLocalStorage only reads at mount, so we mirror the flag in state and react to the
  // dismissal event — otherwise our popover wouldn't appear until this component remounts.
  const [hasSeenDetectIssuesGuidance, setHasSeenDetectIssuesGuidance] = useState(() =>
    getLocalStorageItemByParams({
      key: DEFAULT_DETECT_ISSUES_GUIDANCE_STORAGE_KEY,
      version: DETECT_ISSUES_GUIDANCE_STORAGE_VERSION,
      initialValue: false,
    }),
  );
  useEffect(() => {
    const onDismissed = () => setHasSeenDetectIssuesGuidance(true);
    window.addEventListener(DETECT_ISSUES_GUIDANCE_DISMISSED_EVENT, onDismissed);
    return () => window.removeEventListener(DETECT_ISSUES_GUIDANCE_DISMISSED_EVENT, onDismissed);
  }, []);
  const detectIssuesGuidancePending = shouldEnableIssueDetection() && !hasSeenDetectIssuesGuidance;
  const showGuidance = !hasSeenGuidance && !detectIssuesGuidancePending;

  const handleDismissGuidance = () => {
    setHasSeenGuidance(true);
  };

  const buttonContent = (
    <Button
      componentId={componentId}
      onClick={onClick}
      icon={<SparkleIcon color="ai" />}
      // data-assistant-ui marks this as assistant UI so AssistantAwareDrawer won't treat the
      // click as an outside-click and close. See AssistantAwareDrawer.tsx.
      data-assistant-ui="true"
      aria-label={intl.formatMessage({
        defaultMessage: 'Analyze traces with the MLflow assistant',
        description: 'Aria label for the analyze with assistant button',
      })}
      css={{
        // AI-gradient border on the page surface signals this is an AI-powered action,
        // matching the Detect Issues button and the assistant floating button.
        border: '1px solid transparent !important',
        background: `linear-gradient(${theme.colors.backgroundPrimary}, ${theme.colors.backgroundPrimary}) padding-box, ${theme.gradients.aiBorderGradient} border-box`,
      }}
    >
      <FormattedMessage
        defaultMessage="Analyze with Assistant"
        description="Traces table toolbar button that opens the MLflow assistant to analyze the current traces"
      />
    </Button>
  );

  if (!showGuidance) {
    return <div>{buttonContent}</div>;
  }

  return (
    <Popover.Root
      componentId="mlflow.assistant.traces_guidance"
      open={showGuidance}
      onOpenChange={(open) => {
        // Only allow closing via explicit dismiss actions, not by clicking outside.
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
              defaultMessage="Chat with Traces in Assistant"
              description="Analyze with assistant guidance popover title"
            />
          </div>
          <Button
            componentId="mlflow.assistant.traces_guidance.dismiss"
            icon={<CloseIcon />}
            onClick={handleDismissGuidance}
            aria-label={intl.formatMessage({
              defaultMessage: 'Close guidance',
              description: 'Aria label for closing the analyze with assistant guidance popover',
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
            defaultMessage="Open the MLflow Assistant to ask questions about your traces, debug failures, and investigate quality issues right alongside your data."
            description="Analyze with assistant guidance popover message"
          />
        </div>
        <Button
          componentId="mlflow.assistant.traces_guidance.got_it"
          onClick={handleDismissGuidance}
          css={{ marginTop: theme.spacing.md, width: '100%' }}
        >
          <FormattedMessage defaultMessage="Got it" description="Analyze with assistant guidance dismiss button" />
        </Button>
      </Popover.Content>
    </Popover.Root>
  );
};
