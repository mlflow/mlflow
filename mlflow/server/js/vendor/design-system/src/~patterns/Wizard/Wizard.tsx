import { isUndefined } from 'lodash';
import { useState } from 'react';

import { HorizontalWizardContent } from './HorizontalWizardContent';
import { VerticalWizardContent } from './VerticalWizardContent';
import type { WizardControlledProps, WizardProps } from './WizardProps';
import { useWizardCurrentStep } from './useWizardCurrentStep';

export { MAX_VERTICAL_WIZARD_CONTENT_WIDTH, FIXED_VERTICAL_STEPPER_WIDTH } from './VerticalWizardContent';

export function Wizard({ steps, onStepChanged, onValidateStepChange, initialStep, ...props }: WizardProps) {
  const [currentStepIndex, setCurrentStepIndex] = useState<number>(initialStep ?? 0);
  const currentStepProps = useWizardCurrentStep({
    currentStepIndex,
    setCurrentStepIndex,
    totalSteps: steps.length,
    onStepChanged,
    onValidateStepChange,
  });

  return (
    <WizardControlled
      {...currentStepProps}
      currentStepIndex={currentStepIndex}
      initialStep={initialStep}
      steps={steps}
      onStepChanged={onStepChanged}
      {...props}
    />
  );
}

export function WizardControlled({
  initialStep = 0,
  layout = 'vertical',
  width = '100%',
  height = '100%',
  steps,
  title,
  ...restOfProps
}: WizardControlledProps) {
  if (steps.length === 0 || (!isUndefined(initialStep) && (initialStep < 0 || initialStep >= steps.length))) {
    return null;
  }

  if (layout === 'vertical') {
    return <VerticalWizardContent width={width} height={height} steps={steps} title={title} {...restOfProps} />;
  }

  return <HorizontalWizardContent width={width} height={height} steps={steps} {...restOfProps} />;
}
