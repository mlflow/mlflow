import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { isUndefined } from 'lodash';
import { useState } from 'react';
import { HorizontalWizardStepsContent } from './HorizontalWizardStepsContent';
import { getWizardFooterButtons } from './WizardFooter';
import { useWizardCurrentStep } from './useWizardCurrentStep';
import { Modal } from '../../design-system';
export function WizardModal({ onStepChanged, onCancel, initialStep, steps, onModalClose, localizeStepNumber, cancelButtonContent, nextButtonContent, previousButtonContent, doneButtonContent, enableClickingToSteps, ...modalProps }) {
    const [currentStepIndex, setCurrentStepIndex] = useState(initialStep ?? 0);
    const { onStepsChange, isLastStep, ...footerActions } = useWizardCurrentStep({
        currentStepIndex,
        setCurrentStepIndex,
        totalSteps: steps.length,
        onStepChanged,
    });
    if (steps.length === 0 || (!isUndefined(initialStep) && (initialStep < 0 || initialStep >= steps.length))) {
        return null;
    }
    const footerButtons = getWizardFooterButtons({
        onCancel,
        isLastStep,
        currentStepIndex,
        doneButtonContent,
        previousButtonContent,
        nextButtonContent,
        cancelButtonContent,
        ...footerActions,
        ...steps[currentStepIndex],
    });
    return (_jsx(Modal, { ...modalProps, onCancel: onModalClose, size: "wide", footer: footerButtons, children: _jsx(HorizontalWizardStepsContent, { steps: steps, currentStepIndex: currentStepIndex, localizeStepNumber: localizeStepNumber, enableClickingToSteps: Boolean(enableClickingToSteps), goToStep: footerActions.goToStep }) }));
}
//# sourceMappingURL=WizardModal.js.map