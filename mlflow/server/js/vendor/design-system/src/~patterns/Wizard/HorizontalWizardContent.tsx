import { HorizontalWizardStepsContent } from './HorizontalWizardStepsContent';
import type { WizardFooterProps } from './WizardFooter';
import { getWizardFooterButtons } from './WizardFooter';
import type { WizardControlledProps } from './WizardProps';
import { Spacer, useDesignSystemTheme } from '../../design-system';
import { addDebugOutlineIfEnabled } from '../../design-system/utils/debug';

type HorizontalWizardContentProps = Omit<WizardControlledProps, 'layout' | 'title' | 'initialStep'>;

export function HorizontalWizardContent({
  width,
  height,
  steps,
  currentStepIndex,
  localizeStepNumber,
  onStepsChange,
  enableClickingToSteps,
  hideDescriptionForFutureSteps,
  ...footerProps
}: HorizontalWizardContentProps) {
  return (
    <div
      css={{
        width,
        height,
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'flex-start',
      }}
      {...addDebugOutlineIfEnabled()}
    >
      <HorizontalWizardStepsContent
        steps={steps}
        currentStepIndex={currentStepIndex}
        localizeStepNumber={localizeStepNumber}
        enableClickingToSteps={Boolean(enableClickingToSteps)}
        goToStep={footerProps.goToStep}
        hideDescriptionForFutureSteps={hideDescriptionForFutureSteps}
      />
      <Spacer size="lg" />
      <WizardFooter
        currentStepIndex={currentStepIndex}
        {...steps[currentStepIndex]}
        {...footerProps}
        moveCancelToOtherSide
      />
    </div>
  );
}

function WizardFooter(props: WizardFooterProps) {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'row',
        justifyContent: 'flex-end',
        columnGap: theme.spacing.sm,
        paddingTop: theme.spacing.md,
        paddingBottom: theme.spacing.md,
        borderTop: `1px solid ${theme.colors.border}`,
      }}
    >
      {getWizardFooterButtons(props)}
    </div>
  );
}
