import { isUndefined } from 'lodash';
import { useState } from 'react';

import { HorizontalWizardStepsContent } from './HorizontalWizardStepsContent';
import { getWizardFooterButtons } from './WizardFooter';
import type { WizardProps } from './WizardProps';
import { useWizardCurrentStep } from './useWizardCurrentStep';
import { Modal, type ModalProps } from '../../design-system';

export interface WizardModalProps
  extends Omit<ModalProps, 'footer' | 'onCancel' | 'onOk'>,
    Omit<WizardProps, 'width' | 'layout' | 'title'> {
  onModalClose: () => void;
}

export function WizardModal({
  onStepChanged,
  onCancel,
  initialStep,
  steps,
  onModalClose,
  localizeStepNumber,
  cancelButtonContent,
  nextButtonContent,
  previousButtonContent,
  doneButtonContent,
  enableClickingToSteps,
  ...modalProps
}: WizardModalProps) {
  const [currentStepIndex, setCurrentStepIndex] = useState<number>(initialStep ?? 0);
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

  return (
    <Modal {...modalProps} onCancel={onModalClose} size="wide" footer={footerButtons}>
      <HorizontalWizardStepsContent
        steps={steps}
        currentStepIndex={currentStepIndex}
        localizeStepNumber={localizeStepNumber}
        enableClickingToSteps={Boolean(enableClickingToSteps)}
        goToStep={footerActions.goToStep}
      />
    </Modal>
  );
}
