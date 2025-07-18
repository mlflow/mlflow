import type { Interpolation } from '@emotion/react';
import * as RadixTabs from '@radix-ui/react-tabs';
import React from 'react';
import type { ButtonProps } from '..';
import { DesignSystemEventProviderAnalyticsEventTypes } from '..';
import type { Theme } from '../../theme';
import type { AnalyticsEventValueChangeNoPiiFlagProps, DangerousGeneralProps } from '../types';
interface RootProps extends Omit<RadixTabs.TabsProps, 'asChild' | 'orientation' | 'dir'>, AnalyticsEventValueChangeNoPiiFlagProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange | DesignSystemEventProviderAnalyticsEventTypes.OnView> {
}
export declare const Root: React.ForwardRefExoticComponent<RootProps & React.RefAttributes<HTMLDivElement>>;
interface AddButtonProps extends DangerousGeneralProps, Pick<ButtonProps, 'onClick' | 'className'> {
}
interface ListProps extends DangerousGeneralProps, Omit<RadixTabs.TabsListProps, 'asChild' | 'loop'> {
    /** The add tab button is only displayed when this prop is passed. Include the `onClick` handler for the button's action */
    addButtonProps?: AddButtonProps;
    /** Styling for the list's scrollable area viewport */
    scrollAreaViewportCss?: Interpolation<Theme>;
    /** If using a background color other than primary, pass this color along so the shadow scroll styling can use this color*/
    shadowScrollStylesBackgroundColor?: string;
    /** For customizing the height of the list's scrollbar. The default height is 3px.
     *  Customizing the scrollbar height is not recommended. Ask in #dubois before using.
     */
    scrollbarHeight?: number;
    /** Optional callback to get access to the viewport element. */
    getScrollAreaViewportRef?: (element: HTMLDivElement | null) => void;
    /** Optional CSS to append to the tab list container */
    tabListCss?: Interpolation<Theme>;
}
export declare const List: React.ForwardRefExoticComponent<ListProps & React.RefAttributes<HTMLDivElement>>;
export interface TriggerProps extends Omit<RadixTabs.TabsTriggerProps, 'asChild'> {
    /** Called when the close tab icon is clicked. The close icon is only displayed when this prop is passed */
    onClose?: (value: string) => void;
    /** Disallow 'Delete' key to close the tab.
     * Suppressing the Delete close behavior is not recommended and likely requires additional changes on the calling side to support closing tabs via keyboard navigation.
     * Ask in #dubois before using. */
    suppressDeleteClose?: boolean;
    customizedCloseAriaLabel?: string;
}
export declare const Trigger: React.ForwardRefExoticComponent<TriggerProps & React.RefAttributes<HTMLButtonElement>>;
export declare const Content: React.ForwardRefExoticComponent<Omit<RadixTabs.TabsContentProps, "asChild"> & React.RefAttributes<HTMLDivElement>>;
export {};
//# sourceMappingURL=Tabs.d.ts.map