import type { Interpolation } from '@emotion/react';
import * as RadixTabs from '@radix-ui/react-tabs';
import React from 'react';
import type { ButtonProps } from '../../design-system';
import type { DangerousGeneralProps } from '../../design-system/types';
import type { Theme } from '../../theme';
export declare const Root: React.ForwardRefExoticComponent<Omit<RadixTabs.TabsProps, "orientation" | "dir" | "asChild" | "activationMode"> & React.RefAttributes<HTMLDivElement>>;
interface AddButtonProps extends DangerousGeneralProps, Pick<ButtonProps, 'onClick' | 'componentId' | 'className'> {
}
interface ListProps extends DangerousGeneralProps, Omit<RadixTabs.TabsListProps, 'asChild' | 'loop'> {
    /** The add tab button is only displayed when this prop is passed. Include the `onClick` handler for the button's action */
    addButtonProps?: AddButtonProps;
    /** Styling for the list's scrollable area viewport */
    scrollAreaViewportCss?: Interpolation<Theme>;
}
export declare const List: React.ForwardRefExoticComponent<ListProps & React.RefAttributes<HTMLDivElement>>;
interface TriggerProps extends Omit<RadixTabs.TabsTriggerProps, 'asChild'> {
    /** Called when the close tab icon is clicked. The close icon is only displayed when this prop is passed */
    onClose?: (value: string) => void;
}
export declare const Trigger: React.ForwardRefExoticComponent<TriggerProps & React.RefAttributes<HTMLButtonElement>>;
export declare const Content: React.ForwardRefExoticComponent<Omit<RadixTabs.TabsContentProps, "asChild"> & React.RefAttributes<HTMLDivElement>>;
export {};
//# sourceMappingURL=TabsV2.d.ts.map