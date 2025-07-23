import { jsx as _jsx, Fragment as _Fragment, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { useStepperStepsFromWizardSteps } from './useStepperStepsFromWizardSteps';
import { getShadowScrollStyles, useDesignSystemTheme } from '../../design-system';
import { Stepper } from '../../design-system/Stepper';
export function HorizontalWizardStepsContent({ steps: wizardSteps, currentStepIndex, localizeStepNumber, enableClickingToSteps, goToStep, hideDescriptionForFutureSteps = false, }) {
    const { theme } = useDesignSystemTheme();
    const stepperSteps = useStepperStepsFromWizardSteps(wizardSteps, currentStepIndex, hideDescriptionForFutureSteps);
    const expandContentToFullHeight = wizardSteps[currentStepIndex].expandContentToFullHeight ?? true;
    const disableDefaultScrollBehavior = wizardSteps[currentStepIndex].disableDefaultScrollBehavior ?? false;
    return (_jsxs(_Fragment, { children: [_jsx(Stepper, { currentStepIndex: currentStepIndex, direction: "horizontal", localizeStepNumber: localizeStepNumber, steps: stepperSteps, responsive: false, onStepClicked: enableClickingToSteps ? goToStep : undefined }), _jsx("div", { css: {
                    marginTop: theme.spacing.md,
                    flexGrow: expandContentToFullHeight ? 1 : undefined,
                    overflowY: disableDefaultScrollBehavior ? 'hidden' : 'auto',
                    ...(!disableDefaultScrollBehavior ? getShadowScrollStyles(theme) : {}),
                }, children: wizardSteps[currentStepIndex].content })] }));
}
//# sourceMappingURL=HorizontalWizardStepsContent.js.map