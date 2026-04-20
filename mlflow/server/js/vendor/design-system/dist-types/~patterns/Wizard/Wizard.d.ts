import type { WizardControlledProps, WizardProps } from './WizardProps';
export { MAX_VERTICAL_WIZARD_CONTENT_WIDTH, FIXED_VERTICAL_STEPPER_WIDTH } from './VerticalWizardContent';
export declare function Wizard({ steps, onStepChanged, onValidateStepChange, initialStep, ...props }: WizardProps): import("@emotion/react/jsx-runtime").JSX.Element;
export declare function WizardControlled({ initialStep, layout, width, height, steps, title, ...restOfProps }: WizardControlledProps): import("@emotion/react/jsx-runtime").JSX.Element | null;
//# sourceMappingURL=Wizard.d.ts.map