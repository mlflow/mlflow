/**
 * Setup wizard for MLflow Assistant.
 * Guides users through the setup process.
 */

import { useState, useCallback } from 'react';
import { FormattedMessage } from 'react-intl';
import { Typography, useDesignSystemTheme, CheckCircleIcon } from '@databricks/design-system';

import type { SetupStep, AuthState } from '../types';
import { SetupStepProvider } from './SetupStepProvider';
import { SetupStepAuth } from './SetupStepAuth';
import { SetupStepProject } from './SetupStepProject';
import { SetupComplete } from './SetupComplete';

interface StepIndicatorProps {
  currentStep: SetupStep;
  completedSteps: Set<SetupStep>;
}

const STEPS: { key: SetupStep; label: string }[] = [
  { key: 'provider', label: 'Provider' },
  { key: 'connection', label: 'Connection' },
  { key: 'project', label: 'Project' },
];

const StepIndicator = ({ currentStep, completedSteps }: StepIndicatorProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        justifyContent: 'space-between',
        padding: `${theme.spacing.md}px 0`,
        marginBottom: theme.spacing.lg,
        borderBottom: `1px solid ${theme.colors.border}`,
      }}
    >
      {STEPS.map((step, index) => {
        const isCompleted = completedSteps.has(step.key);
        const isCurrent = currentStep === step.key;
        const isPast = STEPS.findIndex((s) => s.key === currentStep) > index;

        return (
          <div
            key={step.key}
            css={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              flex: 1,
            }}
          >
            <div
              css={{
                width: 24,
                height: 24,
                borderRadius: '50%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                backgroundColor:
                  isCompleted || isPast || isCurrent
                    ? theme.colors.actionPrimaryBackgroundDefault
                    : theme.colors.backgroundSecondary,
                border: isCurrent ? `2px solid ${theme.colors.actionPrimaryBackgroundDefault}` : 'none',
                color:
                  isCompleted || isPast || isCurrent
                    ? theme.colors.actionPrimaryTextDefault
                    : theme.colors.textSecondary,
                fontSize: theme.typography.fontSizeSm,
                fontWeight: theme.typography.typographyBoldFontWeight,
              }}
            >
              {isCompleted || isPast ? <CheckCircleIcon css={{ fontSize: 16 }} /> : index + 1}
            </div>
            <Typography.Text
              color={isCurrent ? 'primary' : 'secondary'}
              css={{
                marginTop: theme.spacing.xs,
                fontSize: theme.typography.fontSizeSm,
                fontWeight: isCurrent ? theme.typography.typographyBoldFontWeight : 'normal',
              }}
            >
              {step.label}
            </Typography.Text>
          </div>
        );
      })}
    </div>
  );
};

interface AssistantSetupWizardProps {
  experimentId?: string;
  onComplete: () => void;
  /** Initial step to start at (for settings flow) */
  initialStep?: SetupStep;
  /** Callback for back button when in settings mode */
  onBack?: () => void;
}

export const AssistantSetupWizard = ({
  experimentId,
  onComplete,
  initialStep,
  onBack: onBackFromSettings,
}: AssistantSetupWizardProps) => {
  const { theme } = useDesignSystemTheme();

  // Settings mode: when we start at a specific step (not 'provider')
  const isSettingsMode = initialStep && initialStep !== 'provider';

  const [currentStep, setCurrentStep] = useState<SetupStep>(initialStep || 'provider');
  const [completedSteps, setCompletedSteps] = useState<Set<SetupStep>>(new Set());
  const [selectedProvider, setSelectedProvider] = useState<string>('claude_code');
  const [cachedAuthStatus, setCachedAuthStatus] = useState<Record<string, AuthState>>({});

  const markStepComplete = useCallback((step: SetupStep) => {
    setCompletedSteps((prev) => new Set([...prev, step]));
  }, []);

  const handleProviderContinue = useCallback(
    (provider: string) => {
      setSelectedProvider(provider);
      markStepComplete('provider');
      setCurrentStep('connection');
    },
    [markStepComplete],
  );

  const handleConnectionContinue = useCallback(() => {
    markStepComplete('connection');
    setCurrentStep('project');
  }, [markStepComplete]);

  const handleConnectionStatusChange = useCallback((provider: string, status: AuthState) => {
    setCachedAuthStatus((prev) => ({ ...prev, [provider]: status }));
  }, []);

  const handleProjectComplete = useCallback(() => {
    markStepComplete('project');
    setCurrentStep('complete');
  }, [markStepComplete]);

  const handleBack = useCallback(() => {
    const stepIndex = STEPS.findIndex((s) => s.key === currentStep);
    if (stepIndex > 0) {
      // Remove current step from completed when going back
      setCompletedSteps((prev) => {
        const next = new Set(prev);
        next.delete(currentStep);
        return next;
      });
      setCurrentStep(STEPS[stepIndex - 1].key);
    }
  }, [currentStep]);

  const renderStepContent = () => {
    switch (currentStep) {
      case 'provider':
        return <SetupStepProvider selectedProvider={selectedProvider} onContinue={handleProviderContinue} />;
      case 'connection':
        return (
          <SetupStepAuth
            provider={selectedProvider}
            cachedAuthStatus={cachedAuthStatus[selectedProvider]}
            onAuthStatusChange={(status) => handleConnectionStatusChange(selectedProvider, status)}
            onBack={handleBack}
            onContinue={handleConnectionContinue}
          />
        );
      case 'project':
        return (
          <SetupStepProject
            experimentId={experimentId}
            onBack={isSettingsMode && onBackFromSettings ? onBackFromSettings : handleBack}
            onComplete={handleProjectComplete}
            nextLabel={isSettingsMode ? 'Save' : 'Finish'}
            backLabel={isSettingsMode ? 'Cancel' : 'Back'}
          />
        );
      case 'complete':
        return <SetupComplete onStartChatting={onComplete} />;
      default:
        return null;
    }
  };

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        padding: theme.spacing.lg,
        overflow: 'auto',
      }}
    >
      <Typography.Title level={4} css={{ marginBottom: theme.spacing.sm }}>
        {isSettingsMode ? (
          <FormattedMessage defaultMessage="Settings" description="Title for the MLflow Assistant settings wizard" />
        ) : (
          <FormattedMessage
            defaultMessage="Setup MLflow Assistant"
            description="Title for the MLflow Assistant setup wizard"
          />
        )}
      </Typography.Title>

      {!isSettingsMode && currentStep !== 'complete' && (
        <StepIndicator currentStep={currentStep} completedSteps={completedSteps} />
      )}

      <div css={{ flex: 1, overflow: 'auto' }}>{renderStepContent()}</div>
    </div>
  );
};
