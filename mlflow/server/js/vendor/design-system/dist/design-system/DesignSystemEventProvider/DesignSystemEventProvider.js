import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import React, { createContext, useContext, useMemo } from 'react';
import { useDesignSystemEventSuppressInteractionContext } from './DesignSystemEventSuppressInteractionProvider';
import { useStableUuidV4 } from '../utils/useStableUuidV4';
export var DesignSystemEventProviderComponentTypes;
(function (DesignSystemEventProviderComponentTypes) {
    DesignSystemEventProviderComponentTypes["Accordion"] = "accordion";
    DesignSystemEventProviderComponentTypes["Alert"] = "alert";
    DesignSystemEventProviderComponentTypes["Banner"] = "banner";
    DesignSystemEventProviderComponentTypes["Button"] = "button";
    DesignSystemEventProviderComponentTypes["Card"] = "card";
    DesignSystemEventProviderComponentTypes["Checkbox"] = "checkbox";
    DesignSystemEventProviderComponentTypes["ContextMenuCheckboxItem"] = "context_menu_checkbox_item";
    DesignSystemEventProviderComponentTypes["ContextMenuItem"] = "context_menu_item";
    DesignSystemEventProviderComponentTypes["ContextMenuRadioGroup"] = "context_menu_radio_group";
    DesignSystemEventProviderComponentTypes["DialogCombobox"] = "dialog_combobox";
    DesignSystemEventProviderComponentTypes["Drawer"] = "drawer_content";
    DesignSystemEventProviderComponentTypes["DropdownMenuCheckboxItem"] = "dropdown_menu_checkbox_item";
    DesignSystemEventProviderComponentTypes["DropdownMenuItem"] = "dropdown_menu_item";
    DesignSystemEventProviderComponentTypes["DropdownMenuRadioGroup"] = "dropdown_menu_radio_group";
    DesignSystemEventProviderComponentTypes["Form"] = "form";
    DesignSystemEventProviderComponentTypes["Input"] = "input";
    DesignSystemEventProviderComponentTypes["LegacySelect"] = "legacy_select";
    DesignSystemEventProviderComponentTypes["Listbox"] = "listbox";
    DesignSystemEventProviderComponentTypes["Modal"] = "modal";
    DesignSystemEventProviderComponentTypes["Notification"] = "notification";
    DesignSystemEventProviderComponentTypes["Pagination"] = "pagination";
    DesignSystemEventProviderComponentTypes["PillControl"] = "pill_control";
    DesignSystemEventProviderComponentTypes["Popover"] = "popover";
    DesignSystemEventProviderComponentTypes["PreviewCard"] = "preview_card";
    DesignSystemEventProviderComponentTypes["Radio"] = "radio";
    DesignSystemEventProviderComponentTypes["RadioGroup"] = "radio_group";
    DesignSystemEventProviderComponentTypes["SegmentedControlGroup"] = "segmented_control_group";
    DesignSystemEventProviderComponentTypes["SimpleSelect"] = "simple_select";
    DesignSystemEventProviderComponentTypes["Switch"] = "switch";
    DesignSystemEventProviderComponentTypes["TableHeader"] = "table_header";
    DesignSystemEventProviderComponentTypes["Tabs"] = "tabs";
    DesignSystemEventProviderComponentTypes["Tag"] = "tag";
    DesignSystemEventProviderComponentTypes["TextArea"] = "text_area";
    DesignSystemEventProviderComponentTypes["ToggleButton"] = "toggle_button";
    DesignSystemEventProviderComponentTypes["Tooltip"] = "tooltip";
    DesignSystemEventProviderComponentTypes["TypeaheadCombobox"] = "typeahead_combobox";
    DesignSystemEventProviderComponentTypes["TypographyLink"] = "typography_link";
})(DesignSystemEventProviderComponentTypes || (DesignSystemEventProviderComponentTypes = {}));
export var DesignSystemEventProviderAnalyticsEventTypes;
(function (DesignSystemEventProviderAnalyticsEventTypes) {
    DesignSystemEventProviderAnalyticsEventTypes["OnClick"] = "onClick";
    DesignSystemEventProviderAnalyticsEventTypes["OnSubmit"] = "onSubmit";
    DesignSystemEventProviderAnalyticsEventTypes["OnValueChange"] = "onValueChange";
    DesignSystemEventProviderAnalyticsEventTypes["OnView"] = "onView";
})(DesignSystemEventProviderAnalyticsEventTypes || (DesignSystemEventProviderAnalyticsEventTypes = {}));
export var DesignSystemEventProviderComponentSubTypes;
(function (DesignSystemEventProviderComponentSubTypes) {
    DesignSystemEventProviderComponentSubTypes["Success"] = "success";
    DesignSystemEventProviderComponentSubTypes["Error"] = "error";
    DesignSystemEventProviderComponentSubTypes["Warning"] = "warning";
    DesignSystemEventProviderComponentSubTypes["Info"] = "info";
    DesignSystemEventProviderComponentSubTypes["InfoLightPurple"] = "info_light_purple";
    DesignSystemEventProviderComponentSubTypes["InfoDarkPurple"] = "info_dark_purple";
})(DesignSystemEventProviderComponentSubTypes || (DesignSystemEventProviderComponentSubTypes = {}));
export const DesignSystemEventProviderComponentSubTypeMap = {
    success: DesignSystemEventProviderComponentSubTypes.Success,
    error: DesignSystemEventProviderComponentSubTypes.Error,
    warning: DesignSystemEventProviderComponentSubTypes.Warning,
    info: DesignSystemEventProviderComponentSubTypes.Info,
    info_light_purple: DesignSystemEventProviderComponentSubTypes.InfoLightPurple,
    info_dark_purple: DesignSystemEventProviderComponentSubTypes.InfoDarkPurple,
};
const shouldTriggerCallback = (eventType, analyticsEvents) => {
    return analyticsEvents.includes(eventType);
};
const DefaultEmptyCallbacks = { callback: () => { } };
const DesignSystemEventProviderContext = React.createContext(DefaultEmptyCallbacks);
/**
 * This gets the event provider, which is used to pass callbacks to the design system for events such as onClick, onView, and onValueChange.
 * If the value is undefined, then the expected behavior is to not emit any events.
 *
 * @returns DesignSystemEventProviderContextType
 */
const useDesignSystemEventProviderContext = () => {
    return useContext(DesignSystemEventProviderContext);
};
export const ComponentFinderContext = createContext({ dataComponentProps: {} });
export const useComponentFinderContext = (targetComponentType) => {
    const componentFinderContext = useContext(ComponentFinderContext);
    return targetComponentType === componentFinderContext.dataComponentProps['data-component-type']
        ? componentFinderContext.dataComponentProps
        : {};
};
/**
 * NOTE: Manually adding in data-component-* props via spreading dataComponentProps from useDesignSystemEventComponentCallbacks
 * is preferred over direct use of this function. See Button.tsx for an example.
 *
 * The main use case of this function is to avoid adding in extra containers/divs with data-component-* props
 * that may result in unexpected css changes.
 *
 * @returns a react node with injected data-component-* props, if possible
 */
export function augmentWithDataComponentProps(node, dataComponentProps) {
    if (React.isValidElement(node)) {
        return React.cloneElement(node, dataComponentProps);
    }
    if (Array.isArray(node)) {
        return node.map((child) => {
            return augmentWithDataComponentProps(child, dataComponentProps);
        });
    }
    return node;
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
export const useDesignSystemEventComponentCallbacks = ({ componentType, componentId, componentSubType, analyticsEvents, valueHasNoPii, shouldStartInteraction, isInteractionSubject, }) => {
    const context = useDesignSystemEventProviderContext();
    const suppressInteractionContext = useDesignSystemEventSuppressInteractionContext();
    // If shouldStartInteraction is explicitly true or has no suppression when shouldStartInteraction
    // is not defined for an onClick event, then it will start the interaction
    const shouldStartInteractionComplete = useMemo(() => shouldStartInteraction ||
        (shouldStartInteraction === undefined && !suppressInteractionContext.suppressAnalyticsStartInteraction), [suppressInteractionContext, shouldStartInteraction]);
    const componentViewId = useStableUuidV4();
    const callbacks = useMemo(() => {
        const dataComponentProps = componentId
            ? { 'data-component-id': componentId, 'data-component-type': componentType }
            : {};
        if (context === DefaultEmptyCallbacks || componentId === undefined) {
            return {
                onClick: () => { },
                onSubmit: () => { },
                onValueChange: () => { },
                onView: () => { },
                dataComponentProps,
            };
        }
        return {
            onClick: (event) => {
                if (shouldTriggerCallback(DesignSystemEventProviderAnalyticsEventTypes.OnClick, analyticsEvents)) {
                    context.callback({
                        eventType: DesignSystemEventProviderAnalyticsEventTypes.OnClick,
                        componentType,
                        componentId,
                        componentSubType,
                        value: undefined,
                        shouldStartInteraction: shouldStartInteractionComplete,
                        event,
                        isInteractionSubject,
                        componentViewId: componentViewId,
                    });
                }
            },
            onSubmit: (payload) => {
                const mode = shouldTriggerCallback(DesignSystemEventProviderAnalyticsEventTypes.OnSubmit, analyticsEvents)
                    ? 'default'
                    : 'associate_event_only';
                context.callback({
                    eventType: DesignSystemEventProviderAnalyticsEventTypes.OnSubmit,
                    componentType,
                    componentId,
                    componentSubType,
                    value: undefined,
                    shouldStartInteraction: shouldStartInteractionComplete,
                    event: payload.event,
                    mode,
                    referrerComponent: payload.referrerComponent,
                    formPropertyValues: {
                        initial: payload.initialState,
                        final: payload.finalState,
                    },
                });
            },
            onValueChange: (value) => {
                if (shouldTriggerCallback(DesignSystemEventProviderAnalyticsEventTypes.OnValueChange, analyticsEvents)) {
                    context.callback({
                        eventType: DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
                        componentType,
                        componentId,
                        componentSubType,
                        value: valueHasNoPii ? value : undefined,
                        shouldStartInteraction: false,
                        componentViewId: componentViewId,
                    });
                }
            },
            onView: (value) => {
                if (shouldTriggerCallback(DesignSystemEventProviderAnalyticsEventTypes.OnView, analyticsEvents)) {
                    context.callback({
                        eventType: DesignSystemEventProviderAnalyticsEventTypes.OnView,
                        componentType,
                        componentId,
                        componentSubType,
                        value: valueHasNoPii ? value : undefined,
                        shouldStartInteraction: false,
                        componentViewId: componentViewId,
                    });
                }
            },
            dataComponentProps,
        };
    }, [
        context,
        componentId,
        analyticsEvents,
        componentType,
        componentSubType,
        shouldStartInteractionComplete,
        isInteractionSubject,
        valueHasNoPii,
        componentViewId,
    ]);
    return callbacks;
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
export function DesignSystemEventProvider({ children, callback, }) {
    const contextValue = useMemo(() => {
        return { callback };
    }, [callback]);
    return (_jsx(DesignSystemEventProviderContext.Provider, { value: contextValue, children: children }));
}
//# sourceMappingURL=DesignSystemEventProvider.js.map