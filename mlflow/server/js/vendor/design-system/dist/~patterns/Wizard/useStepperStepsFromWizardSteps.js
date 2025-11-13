import { isUndefined, pick } from 'lodash';
import { useMemo } from 'react';
export function useStepperStepsFromWizardSteps(wizardSteps, currentStepIdx, hideDescriptionForFutureSteps) {
    return useMemo(() => wizardSteps.map((wizardStep, stepIdx) => ({
        ...pick(wizardStep, ['title', 'status']),
        description: hideDescriptionForFutureSteps &&
            !(stepIdx <= currentStepIdx ||
                wizardStep.status === 'completed' ||
                wizardStep.status === 'error' ||
                wizardStep.status === 'warning')
            ? undefined
            : wizardStep.description,
        additionalVerticalContent: wizardStep.additionalHorizontalLayoutStepContent,
        clickEnabled: isUndefined(wizardStep.clickEnabled)
            ? isWizardStepEnabled(wizardSteps, stepIdx, currentStepIdx, wizardStep.status)
            : wizardStep.clickEnabled,
    })), [currentStepIdx, hideDescriptionForFutureSteps, wizardSteps]);
}
export function isWizardStepEnabled(steps, stepIdx, currentStepIdx, status) {
    if (stepIdx < currentStepIdx || status === 'completed' || status === 'error' || status === 'warning') {
        return true;
    }
    // if every step before stepIdx is completed then the step is enabled
    return steps.slice(0, stepIdx).every((step) => step.status === 'completed');
}
//# sourceMappingURL=useStepperStepsFromWizardSteps.js.map