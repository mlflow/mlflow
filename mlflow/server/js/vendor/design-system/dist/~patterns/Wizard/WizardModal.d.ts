import type { WizardProps } from './WizardProps';
import { type ModalProps } from '../../design-system';
export interface WizardModalProps extends Omit<ModalProps, 'footer' | 'onCancel' | 'onOk'>, Omit<WizardProps, 'width' | 'layout' | 'title'> {
    onModalClose: () => void;
}
export declare function WizardModal({ onStepChanged, onCancel, initialStep, steps, onModalClose, localizeStepNumber, cancelButtonContent, nextButtonContent, previousButtonContent, doneButtonContent, enableClickingToSteps, ...modalProps }: WizardModalProps): import("@emotion/react/jsx-runtime").JSX.Element | null;
//# sourceMappingURL=WizardModal.d.ts.map