import type { WizardProps } from './Wizard';
import type { ModalProps } from '..';
export interface WizardModalProps extends Omit<ModalProps, 'footer' | 'onCancel' | 'onOk'>, Omit<WizardProps, 'width'> {
    onModalClose: () => void;
}
export declare function WizardModal({ onStepChanged, onCancel, initialStep, steps, onModalClose, localizeStepNumber, cancelButtonContent, nextButtonContent, previousButtonContent, doneButtonContent, ...modalProps }: WizardModalProps): import("@emotion/react/jsx-runtime").JSX.Element | null;
//# sourceMappingURL=WizardModal.d.ts.map