import type { WizardControlledProps } from './WizardProps';
type VerticalWizardContentProps = Omit<WizardControlledProps, 'layout' | 'initialStep'>;
export declare const FIXED_VERTICAL_STEPPER_WIDTH = 280;
export declare const MAX_VERTICAL_WIZARD_CONTENT_WIDTH = 920;
export declare function VerticalWizardContent({ width, height, steps: wizardSteps, currentStepIndex, localizeStepNumber, onStepsChange, title, padding, verticalCompactButtonContent, enableClickingToSteps, verticalDocumentationSidebarConfig, ...footerProps }: VerticalWizardContentProps): import("@emotion/react/jsx-runtime").JSX.Element;
export {};
//# sourceMappingURL=VerticalWizardContent.d.ts.map