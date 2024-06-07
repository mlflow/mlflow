/// <reference types="react" />
import type { PopoverProps } from './Popover';
export interface InfoPopoverProps extends React.HTMLAttributes<HTMLButtonElement> {
    popoverProps?: Omit<PopoverProps, 'children' | 'title'>;
    iconProps?: React.HTMLAttributes<HTMLSpanElement>;
    iconTitle?: string;
    isKeyboardFocusable?: boolean;
    ariaLabel?: string;
}
export declare const InfoPopover: ({ children, popoverProps, iconTitle, iconProps, isKeyboardFocusable, ariaLabel, }: InfoPopoverProps) => JSX.Element;
//# sourceMappingURL=InfoPopover.d.ts.map