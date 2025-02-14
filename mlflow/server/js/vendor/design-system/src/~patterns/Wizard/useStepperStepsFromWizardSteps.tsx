import { isUndefined, pick } from 'lodash';
import { useMemo } from 'react';

import type { WizardStep } from './WizardStep';

export function useStepperStepsFromWizardSteps(
  wizardSteps: WizardStep[],
  currentStepIdx: number,
  hideDescriptionForFutureSteps: boolean,
) {
  return useMemo(
    () =>
      wizardSteps.map((wizardStep, stepIdx) => ({
        ...pick(wizardStep, ['title', 'status']),
        description:
          hideDescriptionForFutureSteps &&
          !(
            stepIdx <= currentStepIdx ||
            wizardStep.status === 'completed' ||
            wizardStep.status === 'error' ||
            wizardStep.status === 'warning'
          )
            ? undefined
            : wizardStep.description,
        additionalVerticalContent: wizardStep.additionalHorizontalLayoutStepContent,
        clickEnabled: isUndefined(wizardStep.clickEnabled)
          ? isWizardStepEnabled(wizardSteps, stepIdx, currentStepIdx, wizardStep.status)
          : wizardStep.clickEnabled,
      })),
    [currentStepIdx, hideDescriptionForFutureSteps, wizardSteps],
  );
}

export function isWizardStepEnabled(
  steps: WizardStep[],
  stepIdx: number,
  currentStepIdx: number,
  status: WizardStep['status'],
): boolean {
  if (stepIdx < currentStepIdx || status === 'completed' || status === 'error' || status === 'warning') {
    return true;
  }

  // if every step before stepIdx is completed then the step is enabled
  return steps.slice(0, stepIdx).every((step) => step.status === 'completed');
}
