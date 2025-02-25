/* eslint-disable react/no-unused-prop-types */
// eslint doesn't like passing props as object through to a function
// disabling to avoid a bunch of duplicate code.

import { compact } from 'lodash';

import type { WizardStep } from './WizardStep';
import type { WizardCurrentStepResult } from './useWizardCurrentStep';
import { Button, Tooltip, type ButtonProps } from '../../design-system';

export interface WizardFooterProps
  extends Omit<WizardCurrentStepResult, 'onStepsChange' | 'onValidateStepChange'>,
    WizardStep {
  currentStepIndex: number;
  onCancel?: () => void;
  moveCancelToOtherSide?: boolean;
  cancelButtonContent: React.ReactNode;
  nextButtonContent: React.ReactNode;
  previousButtonContent: React.ReactNode;
  doneButtonContent: React.ReactNode;
  extraFooterButtonsLeft?: ButtonProps[];
  extraFooterButtonsRight?: ButtonProps[];
  componentId?: string;
}

// Buttons are returned in order with primary button last
export function getWizardFooterButtons({
  title,
  goToNextStepOrDone,
  isLastStep,
  currentStepIndex,
  goToPreviousStep,
  busyValidatingNextStep,
  nextButtonDisabled,
  nextButtonLoading,
  nextButtonContentOverride,
  previousButtonContentOverride,
  previousStepButtonHidden,
  previousButtonDisabled,
  previousButtonLoading,
  cancelButtonContent,
  cancelStepButtonHidden,
  nextButtonContent,
  previousButtonContent,
  doneButtonContent,
  extraFooterButtonsLeft,
  extraFooterButtonsRight,
  onCancel,
  moveCancelToOtherSide,
  componentId,
  tooltipContent,
}: WizardFooterProps) {
  return compact([
    !cancelStepButtonHidden &&
      (moveCancelToOtherSide ? (
        <div css={{ flexGrow: 1 }} key="cancel">
          <CancelButton
            onCancel={onCancel}
            cancelButtonContent={cancelButtonContent}
            componentId={componentId ? `${componentId}.cancel` : undefined}
          />
        </div>
      ) : (
        <CancelButton
          onCancel={onCancel}
          cancelButtonContent={cancelButtonContent}
          componentId={componentId ? `${componentId}.cancel` : undefined}
          key="cancel"
        />
      )),
    currentStepIndex > 0 && !previousStepButtonHidden && (
      <Button
        onClick={goToPreviousStep}
        type="tertiary"
        key="previous"
        disabled={previousButtonDisabled}
        loading={previousButtonLoading}
        componentId={componentId ? `${componentId}.previous` : 'dubois-wizard-footer-previous'}
      >
        {previousButtonContentOverride ? previousButtonContentOverride : previousButtonContent}
      </Button>
    ),
    extraFooterButtonsLeft &&
      extraFooterButtonsLeft.map((buttonProps, index) => (
        <ButtonWithTooltip {...buttonProps} type={undefined} key={`extra-left-${index}`} />
      )),
    <ButtonWithTooltip
      onClick={goToNextStepOrDone}
      disabled={nextButtonDisabled}
      tooltipContent={tooltipContent}
      loading={nextButtonLoading || busyValidatingNextStep}
      type={(extraFooterButtonsRight?.length ?? 0) > 0 ? undefined : 'primary'}
      key="next"
      componentId={componentId ? `${componentId}.next` : 'dubois-wizard-footer-next'}
    >
      {nextButtonContentOverride ? nextButtonContentOverride : isLastStep ? doneButtonContent : nextButtonContent}
    </ButtonWithTooltip>,
    extraFooterButtonsRight &&
      extraFooterButtonsRight.map((buttonProps, index) => (
        <ButtonWithTooltip
          {...buttonProps}
          type={index === extraFooterButtonsRight.length - 1 ? 'primary' : undefined}
          key={`extra-right-${index}`}
        />
      )),
  ]);
}

function CancelButton({
  onCancel,
  cancelButtonContent,
  componentId,
}: Pick<WizardFooterProps, 'onCancel' | 'cancelButtonContent' | 'componentId'>) {
  return (
    <Button onClick={onCancel} type="tertiary" key="cancel" componentId={componentId ?? 'dubois-wizard-footer-cancel'}>
      {cancelButtonContent}
    </Button>
  );
}

export function ButtonWithTooltip({
  tooltipContent,
  disabled,
  children,
  ...props
}: React.ComponentProps<typeof Button> & { tooltipContent?: string }) {
  return tooltipContent ? (
    <Tooltip componentId="dubois-wizard-footer-tooltip" content={tooltipContent}>
      <Button {...props} disabled={disabled}>
        {children}
      </Button>
    </Tooltip>
  ) : (
    <Button {...props} disabled={disabled}>
      {children}
    </Button>
  );
}
