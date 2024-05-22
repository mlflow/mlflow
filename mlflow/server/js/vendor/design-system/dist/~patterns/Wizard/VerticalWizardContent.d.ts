import type { WizardControlledProps } from './WizardProps';
type VerticalWizardContentProps = Omit<WizardControlledProps, 'layout' | 'initialStep'>;
export declare function VerticalWizardContent({ width, height, steps: wizardSteps, currentStepIndex, localizeStepNumber, onStepsChange, expandContentToFullHeight, title, useCustomScrollBehavior, padding, ...footerProps }: VerticalWizardContentProps): import("@emotion/react/jsx-runtime").JSX.Element;
export {};
//# sourceMappingURL=VerticalWizardContent.d.ts.map