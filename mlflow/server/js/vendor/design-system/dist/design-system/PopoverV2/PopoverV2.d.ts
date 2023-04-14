/// <reference types="react" />
import * as Popover from '@radix-ui/react-popover';
export declare const Root: import("react").FC<Popover.PopoverProps>;
export interface PopoverV2Props extends Popover.PopoverContentProps {
    minWidth?: number;
}
export declare const Content: import("react").ForwardRefExoticComponent<PopoverV2Props & import("react").RefAttributes<HTMLDivElement>>;
export declare const Trigger: import("react").ForwardRefExoticComponent<Popover.PopoverTriggerProps & import("react").RefAttributes<HTMLButtonElement>>;
export declare const Close: import("react").ForwardRefExoticComponent<Popover.PopoverCloseProps & import("react").RefAttributes<HTMLButtonElement>>;
export declare const Arrow: import("react").ForwardRefExoticComponent<Popover.PopoverArrowProps & import("react").RefAttributes<SVGSVGElement>>;
//# sourceMappingURL=PopoverV2.d.ts.map