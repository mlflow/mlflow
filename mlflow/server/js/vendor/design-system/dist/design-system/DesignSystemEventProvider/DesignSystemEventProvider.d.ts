import type { UIEvent } from 'react';
import React from 'react';
export type DesignSystemEventTypeMapping<V> = {
    [K in DesignSystemEventProviderAnalyticsEventTypes]: V;
};
export declare enum DesignSystemEventProviderComponentTypes {
    Accordion = "accordion",
    Alert = "alert",
    Banner = "banner",
    Button = "button",
    Card = "card",
    Checkbox = "checkbox",
    ContextMenuCheckboxItem = "context_menu_checkbox_item",
    ContextMenuItem = "context_menu_item",
    ContextMenuRadioGroup = "context_menu_radio_group",
    DialogCombobox = "dialog_combobox",
    Drawer = "drawer_content",
    DropdownMenuCheckboxItem = "dropdown_menu_checkbox_item",
    DropdownMenuItem = "dropdown_menu_item",
    DropdownMenuRadioGroup = "dropdown_menu_radio_group",
    Input = "input",
    LegacySelect = "legacy_select",
    Modal = "modal",
    Notification = "notification",
    Pagination = "pagination",
    PillControl = "pill_control",
    PreviewCard = "preview_card",
    RadioGroup = "radio_group",
    SegmentedControlGroup = "segmented_control_group",
    SimpleSelect = "simple_select",
    Switch = "switch",
    TableHeader = "table_header",
    TabsV2 = "tabs",
    Tag = "tag",
    TextArea = "text_area",
    ToggleButton = "toggle_button",
    Tooltip = "tooltip",
    TypographyLink = "typography_link"
}
export declare enum DesignSystemEventProviderAnalyticsEventTypes {
    OnClick = "onClick",
    OnView = "onView",
    OnValueChange = "onValueChange"
}
export type DesignSystemEventProviderContextType = {
    callback: DesignSystemEventProviderCallback;
};
export type DesignSystemEventProviderCallbackParams = {
    eventType: DesignSystemEventProviderAnalyticsEventTypes;
    componentType: DesignSystemEventProviderComponentTypes;
    componentId: string;
    value: unknown;
    shouldStartInteraction?: boolean;
    event?: UIEvent;
    skip?: boolean;
};
export type DesignSystemEventProviderCallback = (params: DesignSystemEventProviderCallbackParams) => void;
/**
 * NOTE: This is not suggested for direct usage from engineers, and should emit your own events.
 * See https://databricks.atlassian.net/wiki/spaces/UN/pages/2533556277/Usage+Logging+in+UI#Send-usage-logging-from-UI for more details.
 *
 * This gets the event provider component event type callbacks.
 * If context & componentId are not undefined, then it will check if the event is in analyticsEvents or the default analyticsEvents for that component.
 * If the context or componentId are undefined, then the expected behavior is to not emit any events.
 *
 * @returns Object of event callbacks
 */
export declare const useDesignSystemEventComponentCallbacks: ({ componentType, componentId, analyticsEvents, valueHasNoPii, shouldStartInteraction, }: {
    componentType: DesignSystemEventProviderComponentTypes;
    componentId: string | undefined;
    analyticsEvents: ReadonlyArray<DesignSystemEventProviderAnalyticsEventTypes>;
    valueHasNoPii?: boolean;
    shouldStartInteraction?: boolean;
}) => {
    onClick: (event?: UIEvent) => void;
    onValueChange: (value?: any) => void;
    onView: () => void;
};
/**
 * NOTE: This is not suggested for direct usage from engineers, and should use RecordEventContext instead.
 * See https://databricks.atlassian.net/wiki/spaces/UN/pages/2533556277/Usage+Logging+in+UI#Send-usage-logging-from-UI for more details.
 *
 * This is the Design System Event Context Provider, and is only used to pass callbacks to the design system for events such as onClick, onView, and onValueChange.
 *
 * @param children Children react elements
 * @param callback The event callback function
 * @returns Design System Event Context Provider with the children inside
 */
export declare function DesignSystemEventProvider({ children, callback, }: React.PropsWithChildren<DesignSystemEventProviderContextType>): import("@emotion/react/jsx-runtime").JSX.Element;
//# sourceMappingURL=DesignSystemEventProvider.d.ts.map