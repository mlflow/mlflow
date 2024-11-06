import * as Popover from '@radix-ui/react-popover';
import type { ReactElement } from 'react';
import { DesignSystemEventProviderAnalyticsEventTypes } from '../DesignSystemEventProvider';
import type { AnalyticsEventProps } from '../types';
export declare const Anchor: import("react").ForwardRefExoticComponent<Popover.PopoverAnchorProps & import("react").RefAttributes<HTMLDivElement>>;
export type PopoverRootProps = Popover.PopoverProps & AnalyticsEventProps<DesignSystemEventProviderAnalyticsEventTypes.OnView>;
export declare const Root: ({ children, onOpenChange, componentId, analyticsEvents, ...props }: PopoverRootProps) => ReactElement;
export interface PopoverProps extends Popover.PopoverContentProps {
    minWidth?: number;
    maxWidth?: number;
}
export declare const Content: import("react").ForwardRefExoticComponent<PopoverProps & import("react").RefAttributes<HTMLDivElement>>;
export declare const Trigger: import("react").ForwardRefExoticComponent<Popover.PopoverTriggerProps & import("react").RefAttributes<HTMLButtonElement>>;
export declare const Close: import("react").ForwardRefExoticComponent<Popover.PopoverCloseProps & import("react").RefAttributes<HTMLButtonElement>>;
export declare const Arrow: import("react").ForwardRefExoticComponent<Popover.PopoverArrowProps & import("react").RefAttributes<SVGSVGElement>>;
//# sourceMappingURL=Popover.d.ts.map