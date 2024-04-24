/// <reference types="react" />
import type { ModalProps as AntDModalProps } from 'antd';
import type { ButtonProps } from '../Button';
import type { DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
export interface ModalProps extends HTMLDataAttributes, DangerouslySetAntdProps<AntDModalProps> {
    /** Whether or not the modal is currently open. Use together with onOk and onCancel to control the modal state. */
    visible?: AntDModalProps['visible'];
    /** Function called when the primary button is clicked */
    onOk?: AntDModalProps['onOk'];
    /** Function called when the secondary button is clicked */
    onCancel?: AntDModalProps['onCancel'];
    /** Title displayed at the top of the modal */
    title?: AntDModalProps['title'];
    /** Contents displayed in the body of the modal */
    children?: React.ReactNode;
    /** A custom JSX element to render in place of the default footer */
    footer?: AntDModalProps['footer'];
    /** Sets the horizontal size according to the size presets */
    size?: 'normal' | 'wide';
    /** Defines the modal height either by both its content and maximum height (dynamic) or by its maximum height only (maxed_out) */
    verticalSizing?: 'dynamic' | 'maxed_out';
    /** When set to true, the confirm button will show a loading indicator */
    confirmLoading?: AntDModalProps['confirmLoading'];
    /** Function called after the modal has finished closing */
    afterClose?: AntDModalProps['afterClose'];
    /** Text of the primary button */
    okText?: AntDModalProps['okText'];
    /** Text of the secondary button */
    cancelText?: AntDModalProps['cancelText'];
    /** Set to true to force the modal to render */
    forceRender?: AntDModalProps['forceRender'];
    /** When true, the modal content will be destroyed when closed, rather than just visually hidden. */
    destroyOnClose?: AntDModalProps['destroyOnClose'];
    /** A CSS class name to apply to the modal wrapper */
    wrapClassName?: AntDModalProps['wrapClassName'];
    /** A CSS class name to apply to the modal */
    className?: AntDModalProps['className'];
    /** A function that returns the element into which the modal will be rendered */
    getContainer?: AntDModalProps['getContainer'];
    /**
     * Allows setting the CSS z-index of the modal.
     * Can be used to work around ordering issues when the modal is not overlaying other elements correctly
     */
    zIndex?: AntDModalProps['zIndex'];
    /** The OK button props */
    okButtonProps?: Omit<ButtonProps, 'componentId' | 'analyticsEvents'>;
    /** The cancel button props */
    cancelButtonProps?: Omit<ButtonProps, 'componentId' | 'analyticsEvents'>;
    /** Custom modal content render function. While custom rendering is discouraged in general, this prop can be used to wrap the modal contents in a React context. */
    modalRender?: (node: React.ReactNode) => React.ReactNode;
    /** Specify which button to autofocus, only works with default footer */
    autoFocusButton?: 'ok' | 'cancel';
    /** Specify whether to truncate the Modal title when it is too long */
    truncateTitle?: boolean;
}
export declare function Modal({ okButtonProps, cancelButtonProps, dangerouslySetAntdProps, children, title, footer, size, verticalSizing, autoFocusButton, truncateTitle, ...props }: ModalProps): JSX.Element;
export declare function DangerModal(props: Omit<ModalProps, 'footer'>): JSX.Element;
//# sourceMappingURL=Modal.d.ts.map