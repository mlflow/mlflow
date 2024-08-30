import type { DropdownMenuProps } from '@radix-ui/react-dropdown-menu';
import * as React from 'react';
import type { DesignSystemEventProviderAnalyticsEventTypes } from '../../DesignSystemEventProvider/DesignSystemEventProvider';
import type { AnalyticsEventProps } from '../../types';
type SizeType = 'small' | 'middle' | undefined;
interface ButtonGroupProps {
    size?: SizeType;
    style?: React.CSSProperties;
    className?: string;
    prefixCls?: string;
    children?: React.ReactNode;
}
type Placement = 'topLeft' | 'topCenter' | 'topRight' | 'bottomLeft' | 'bottomCenter' | 'bottomRight';
type OverlayFunc = () => React.ReactElement;
type Align = {
    points?: [string, string];
    offset?: [number, number];
    targetOffset?: [number, number];
    overflow?: {
        adjustX?: boolean;
        adjustY?: boolean;
    };
    useCssRight?: boolean;
    useCssBottom?: boolean;
    useCssTransform?: boolean;
};
interface DropdownProps {
    autoFocus?: boolean;
    arrow?: boolean;
    trigger?: ('click' | 'hover' | 'contextMenu')[];
    overlay?: React.ReactElement | OverlayFunc;
    onOpenChange?: (open: boolean) => void;
    open?: boolean;
    disabled?: boolean;
    destroyPopupOnHide?: boolean;
    align?: Align;
    getPopupContainer?: (triggerNode: HTMLElement) => HTMLElement;
    prefixCls?: string;
    className?: string;
    transitionName?: string;
    placement?: Placement;
    overlayClassName?: string;
    overlayStyle?: React.CSSProperties;
    forceRender?: boolean;
    mouseEnterDelay?: number;
    mouseLeaveDelay?: number;
    openClassName?: string;
    children?: React.ReactNode;
    leftButtonIcon?: React.ReactNode;
}
export interface DropdownButtonProps extends ButtonGroupProps, DropdownProps, AnalyticsEventProps<DesignSystemEventProviderAnalyticsEventTypes.OnClick> {
    type?: 'primary';
    htmlType?: 'submit' | 'reset' | 'button';
    danger?: boolean;
    disabled?: boolean;
    loading?: boolean | {
        delay?: number;
    };
    onClick?: React.MouseEventHandler<HTMLButtonElement>;
    icon?: React.ReactNode;
    href?: string;
    children?: React.ReactNode;
    title?: string;
    buttonsRender?: (buttons: [React.ReactNode, React.ReactNode]) => [React.ReactNode, React.ReactNode];
    menuButtonLabel?: string;
    menu?: React.ReactElement;
    dropdownMenuRootProps?: DropdownMenuProps;
    'aria-label'?: string;
}
export declare const DropdownButton: React.FC<DropdownButtonProps>;
export {};
//# sourceMappingURL=DropdownButton.d.ts.map