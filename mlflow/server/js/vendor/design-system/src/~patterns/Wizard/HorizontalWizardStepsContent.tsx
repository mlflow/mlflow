import type { WizardStep } from './WizardStep';
import { useStepperStepsFromWizardSteps } from './useStepperStepsFromWizardSteps';
import { getShadowScrollStyles, useDesignSystemTheme } from '../../design-system';
import type { StepperProps } from '../../development/Stepper';
import { Stepper } from '../../development/Stepper';

export interface HorizontalWizardStepsContentProps {
  steps: WizardStep[];
  currentStepIndex: number;
  localizeStepNumber: StepperProps['localizeStepNumber'];
  enableClickingToSteps: boolean;
  goToStep: (step: number) => void;
  hideDescriptionForFutureSteps?: boolean;
}

export function HorizontalWizardStepsContent({
  steps: wizardSteps,
  currentStepIndex,
  localizeStepNumber,
  enableClickingToSteps,
  goToStep,
  hideDescriptionForFutureSteps = false,
}: HorizontalWizardStepsContentProps) {
  const { theme } = useDesignSystemTheme();
  const stepperSteps = useStepperStepsFromWizardSteps(wizardSteps, currentStepIndex, hideDescriptionForFutureSteps);
  const expandContentToFullHeight = wizardSteps[currentStepIndex].expandContentToFullHeight ?? true;
  const disableDefaultScrollBehavior = wizardSteps[currentStepIndex].disableDefaultScrollBehavior ?? false;

  return (
    <>
      <Stepper
        currentStepIndex={currentStepIndex}
        direction="horizontal"
        localizeStepNumber={localizeStepNumber}
        steps={stepperSteps}
        responsive={false}
        onStepClicked={enableClickingToSteps ? goToStep : undefined}
      />
      <div
        css={{
          marginTop: theme.spacing.md,
          flexGrow: expandContentToFullHeight ? 1 : undefined,

          overflowY: disableDefaultScrollBehavior ? 'hidden' : 'auto',
          ...(!disableDefaultScrollBehavior ? getShadowScrollStyles(theme) : {}),
        }}
      >
        {wizardSteps[currentStepIndex].content}
      </div>
    </>
  );
}
