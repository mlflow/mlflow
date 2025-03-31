import type { FormEvent } from 'react';
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
    Form = "form",
    Input = "input",
    LegacySelect = "legacy_select",
    Modal = "modal",
    Notification = "notification",
    Pagination = "pagination",
    PillControl = "pill_control",
    Popover = "popover",
    PreviewCard = "preview_card",
    Radio = "radio",
    RadioGroup = "radio_group",
    SegmentedControlGroup = "segmented_control_group",
    SimpleSelect = "simple_select",
    Switch = "switch",
    TableHeader = "table_header",
    Tabs = "tabs",
    Tag = "tag",
    TextArea = "text_area",
    ToggleButton = "toggle_button",
    Tooltip = "tooltip",
    TypeaheadCombobox = "typeahead_combobox",
    TypographyLink = "typography_link"
}
export declare enum DesignSystemEventProviderAnalyticsEventTypes {
    OnClick = "onClick",
    OnSubmit = "onSubmit",
    OnValueChange = "onValueChange",
    OnView = "onView"
}
export declare enum DesignSystemEventProviderComponentSubTypes {
    Success = "success",
    Error = "error",
    Warning = "warning",
    Info = "info",
    InfoLightPurple = "info_light_purple",
    InfoDarkPurple = "info_dark_purple"
}
export declare const DesignSystemEventProviderComponentSubTypeMap: {
    success: DesignSystemEventProviderComponentSubTypes.Success;
    error: DesignSystemEventProviderComponentSubTypes.Error;
    warning: DesignSystemEventProviderComponentSubTypes.Warning;
    info: DesignSystemEventProviderComponentSubTypes.Info;
    info_light_purple: DesignSystemEventProviderComponentSubTypes.InfoLightPurple;
    info_dark_purple: DesignSystemEventProviderComponentSubTypes.InfoDarkPurple;
};
export type DesignSystemEventProviderContextType = {
    callback: DesignSystemEventProviderCallback;
};
export type ReferrerComponentType = {
    type: DesignSystemEventProviderComponentTypes;
    id: string;
};
export type DesignSystemEventProviderCallbackParams = {
    eventType: DesignSystemEventProviderAnalyticsEventTypes;
    componentType: DesignSystemEventProviderComponentTypes;
    componentId: string;
    componentSubType?: DesignSystemEventProviderComponentSubTypes;
    value: unknown;
    shouldStartInteraction?: boolean;
    event?: UIEvent | FormEvent;
    mode?: 'default' | 'skip' | 'associate_event_only';
    referrerComponent?: ReferrerComponentType;
    isInteractionSubject?: boolean;
    formPropertyValues?: {
        initial: Record<string, unknown> | undefined;
        final: Record<string, unknown> | undefined;
    };
};
export type DesignSystemEventProviderCallback = (params: DesignSystemEventProviderCallbackParams) => void;
export type DataComponentProps = {
    'data-component-id': string;
    'data-component-type': DesignSystemEventProviderComponentTypes;
} | Record<string, never>;
type ComponentFinderContextProps = {
    dataComponentProps: DataComponentProps;
};
export declare const ComponentFinderContext: React.Context<ComponentFinderContextProps>;
export declare const useComponentFinderContext: (targetComponentType: DesignSystemEventProviderComponentTypes) => DataComponentProps;
/**
 * NOTE: Manually adding in data-component-* props via spreading dataComponentProps from useDesignSystemEventComponentCallbacks
 * is preferred over direct use of this function. See Button.tsx for an example.
 *
 * The main use case of this function is to avoid adding in extra containers/divs with data-component-* props
 * that may result in unexpected css changes.
 *
 * @returns a react node with injected data-component-* props, if possible
 */
export declare function augmentWithDataComponentProps(node: React.ReactNode, dataComponentProps: DataComponentProps): React.ReactNode;
interface OnSubmitParams {
    event: FormEvent;
    initialState: Record<string, unknown> | undefined;
    finalState: Record<string, unknown> | undefined;
    referrerComponent?: ReferrerComponentType;
}
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
export declare const useDesignSystemEventComponentCallbacks: ({ componentType, componentId, componentSubType, analyticsEvents, valueHasNoPii, shouldStartInteraction, isInteractionSubject, }: {
    componentType: DesignSystemEventProviderComponentTypes;
    componentId: string | undefined;
    componentSubType?: DesignSystemEventProviderComponentSubTypes;
    analyticsEvents: ReadonlyArray<DesignSystemEventProviderAnalyticsEventTypes>;
    valueHasNoPii?: boolean;
    shouldStartInteraction?: boolean;
    isInteractionSubject?: boolean;
}) => {
    onClick: (event: React.UIEvent | undefined) => void;
    onSubmit: (payload: OnSubmitParams) => void;
    onValueChange: (value?: any) => void;
    onView: () => void;
    dataComponentProps: DataComponentProps;
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
export {};
//# sourceMappingURL=DesignSystemEventProvider.d.ts.map