import type { ModalProps as AntDModalProps } from 'antd';
import React from 'react';
import type { ButtonProps } from '../Button';
import { DesignSystemEventProviderAnalyticsEventTypes } from '../DesignSystemEventProvider/DesignSystemEventProvider';
import type { AnalyticsEventPropsWithStartInteraction, HTMLDataAttributes } from '../types';
export interface ModalProps extends HTMLDataAttributes, AnalyticsEventPropsWithStartInteraction<DesignSystemEventProviderAnalyticsEventTypes.OnView> {
    dangerouslySetAntdProps?: Partial<Pick<AntDModalProps, 'bodyStyle' | 'width' | 'wrapProps' | 'closable' | 'maskClosable' | 'keyboard' | 'centered' | 'closeIcon' | 'style' | 'mask' | 'modalRender' | 'okType' | 'maskStyle'>>;
    visible?: boolean;
    onOk?: (...args: any[]) => any;
    onCancel?: (...args: any[]) => any;
    title?: React.ReactNode;
    children?: React.ReactNode;
    footer?: React.ReactNode;
    size?: 'normal' | 'wide';
    verticalSizing?: 'dynamic' | 'maxed_out';
    confirmLoading?: boolean;
    afterClose?: () => void;
    okText?: React.ReactNode;
    cancelText?: React.ReactNode;
    destroyOnClose?: boolean;
    wrapClassName?: string;
    className?: string;
    getContainer?: () => HTMLElement | null;
    zIndex?: number;
    okButtonProps?: Omit<ButtonProps, 'componentId' | 'analyticsEvents'>;
    cancelButtonProps?: Omit<ButtonProps, 'componentId' | 'analyticsEvents'>;
    modalRender?: (node: React.ReactNode) => React.ReactNode;
    autoFocusButton?: 'ok' | 'cancel';
    truncateTitle?: boolean;
    /**
     * @default true
     * Prevents auto focusing the first focusable element when the modal is opened */
    preventAutoFocus?: boolean;
}
export declare function Modal(props: ModalProps): JSX.Element;
export declare function DangerModal(props: Omit<ModalProps, 'footer'>): JSX.Element;
//# sourceMappingURL=RadixModal.d.ts.map