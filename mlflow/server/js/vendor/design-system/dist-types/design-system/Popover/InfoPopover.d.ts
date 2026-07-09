import type { PopoverProps } from './Popover';
import type { IconProps } from '../Icon';
export interface InfoPopoverProps extends React.HTMLAttributes<HTMLButtonElement> {
    popoverProps?: Omit<PopoverProps, 'children' | 'title'>;
    iconProps?: IconProps;
    iconTitle?: string;
    isKeyboardFocusable?: boolean;
    ariaLabel?: string;
}
export declare const InfoPopover: ({ children, popoverProps, iconTitle, iconProps, isKeyboardFocusable, ariaLabel, }: InfoPopoverProps) => JSX.Element;
//# sourceMappingURL=InfoPopover.d.ts.map