import type { FormEvent } from 'react';
import React, { createContext, useContext, useMemo } from 'react';

import { useDesignSystemEventSuppressInteractionContext } from './DesignSystemEventSuppressInteractionProvider';

export type DesignSystemEventTypeMapping<V> = { [K in DesignSystemEventProviderAnalyticsEventTypes]: V };

export enum DesignSystemEventProviderComponentTypes {
  Accordion = 'accordion',
  Alert = 'alert',
  Banner = 'banner',
  Button = 'button',
  Card = 'card',
  Checkbox = 'checkbox',
  ContextMenuCheckboxItem = 'context_menu_checkbox_item',
  ContextMenuItem = 'context_menu_item',
  ContextMenuRadioGroup = 'context_menu_radio_group',
  DialogCombobox = 'dialog_combobox',
  Drawer = 'drawer_content',
  DropdownMenuCheckboxItem = 'dropdown_menu_checkbox_item',
  DropdownMenuItem = 'dropdown_menu_item',
  DropdownMenuRadioGroup = 'dropdown_menu_radio_group',
  Form = 'form',
  Input = 'input',
  LegacySelect = 'legacy_select',
  Modal = 'modal',
  Notification = 'notification',
  Pagination = 'pagination',
  PillControl = 'pill_control',
  Popover = 'popover',
  PreviewCard = 'preview_card',
  Radio = 'radio',
  RadioGroup = 'radio_group',
  SegmentedControlGroup = 'segmented_control_group',
  SimpleSelect = 'simple_select',
  Switch = 'switch',
  TableHeader = 'table_header',
  Tabs = 'tabs',
  Tag = 'tag',
  TextArea = 'text_area',
  ToggleButton = 'toggle_button',
  Tooltip = 'tooltip',
  TypeaheadCombobox = 'typeahead_combobox',
  TypographyLink = 'typography_link',
}

export enum DesignSystemEventProviderAnalyticsEventTypes {
  OnClick = 'onClick',
  OnSubmit = 'onSubmit',
  OnValueChange = 'onValueChange',
  OnView = 'onView',
}

export enum DesignSystemEventProviderComponentSubTypes {
  Success = 'success',
  Error = 'error',
  Warning = 'warning',
  Info = 'info',
  InfoLightPurple = 'info_light_purple',
  InfoDarkPurple = 'info_dark_purple',
}

export const DesignSystemEventProviderComponentSubTypeMap: Record<string, DesignSystemEventProviderComponentSubTypes> =
  {
    success: DesignSystemEventProviderComponentSubTypes.Success,
    error: DesignSystemEventProviderComponentSubTypes.Error,
    warning: DesignSystemEventProviderComponentSubTypes.Warning,
    info: DesignSystemEventProviderComponentSubTypes.Info,
    info_light_purple: DesignSystemEventProviderComponentSubTypes.InfoLightPurple,
    info_dark_purple: DesignSystemEventProviderComponentSubTypes.InfoDarkPurple,
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
  skip?: boolean;
  referrerComponent?: ReferrerComponentType;
  isInteractionSubject?: boolean;
};

export type DesignSystemEventProviderCallback = (params: DesignSystemEventProviderCallbackParams) => void;

const shouldTriggerCallback = (
  eventType: DesignSystemEventProviderAnalyticsEventTypes,
  analyticsEvents: ReadonlyArray<DesignSystemEventProviderAnalyticsEventTypes>,
): boolean => {
  return analyticsEvents.includes(eventType);
};

const DefaultEmptyCallbacks: DesignSystemEventProviderContextType = { callback: () => {} };

const DesignSystemEventProviderContext =
  React.createContext<DesignSystemEventProviderContextType>(DefaultEmptyCallbacks);

/**
 * This gets the event provider, which is used to pass callbacks to the design system for events such as onClick, onView, and onValueChange.
 * If the value is undefined, then the expected behavior is to not emit any events.
 *
 * @returns DesignSystemEventProviderContextType
 */
const useDesignSystemEventProviderContext = () => {
  return useContext(DesignSystemEventProviderContext);
};

export type DataComponentProps =
  | {
      'data-component-id': string;
      'data-component-type': DesignSystemEventProviderComponentTypes;
    }
  | Record<string, never>;

type ComponentFinderContextProps = {
  dataComponentProps: DataComponentProps;
};

export const ComponentFinderContext = createContext<ComponentFinderContextProps>({ dataComponentProps: {} });

export const useComponentFinderContext = (
  targetComponentType: DesignSystemEventProviderComponentTypes,
): DataComponentProps => {
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
export function augmentWithDataComponentProps(
  node: React.ReactNode,
  dataComponentProps: DataComponentProps,
): React.ReactNode {
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
export const useDesignSystemEventComponentCallbacks = ({
  componentType,
  componentId,
  componentSubType,
  analyticsEvents,
  valueHasNoPii,
  shouldStartInteraction,
  isInteractionSubject,
}: {
  componentType: DesignSystemEventProviderComponentTypes;
  componentId: string | undefined;
  componentSubType?: DesignSystemEventProviderComponentSubTypes;
  analyticsEvents: ReadonlyArray<DesignSystemEventProviderAnalyticsEventTypes>;
  valueHasNoPii?: boolean;
  shouldStartInteraction?: boolean;
  isInteractionSubject?: boolean;
}): {
  onClick: (event: React.UIEvent | undefined) => void;
  onSubmit: (event: FormEvent, referrerComponent?: ReferrerComponentType) => void;
  onValueChange: (value?: any) => void;
  onView: () => void;
  dataComponentProps: DataComponentProps;
} => {
  const context = useDesignSystemEventProviderContext();
  const suppressInteractionContext = useDesignSystemEventSuppressInteractionContext();
  // If shouldStartInteraction is explicitly true or has no suppression when shouldStartInteraction
  // is not defined for an onClick event, then it will start the interaction
  const shouldStartInteractionComplete = useMemo(
    () =>
      shouldStartInteraction ||
      (shouldStartInteraction === undefined &&
        !suppressInteractionContext.suppressAnalyticsStartInteraction &&
        (shouldTriggerCallback(DesignSystemEventProviderAnalyticsEventTypes.OnClick, analyticsEvents) ||
          shouldTriggerCallback(DesignSystemEventProviderAnalyticsEventTypes.OnSubmit, analyticsEvents))),
    [suppressInteractionContext, shouldStartInteraction, analyticsEvents],
  );

  const callbacks = useMemo(() => {
    const dataComponentProps = componentId
      ? { 'data-component-id': componentId, 'data-component-type': componentType }
      : ({} as DataComponentProps);
    if (context === DefaultEmptyCallbacks || componentId === undefined) {
      return {
        onClick: () => {},
        onSubmit: () => {},
        onValueChange: () => {},
        onView: () => {},
        dataComponentProps,
      };
    }

    return {
      onClick: (event: React.UIEvent | undefined) => {
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
          });
        }
      },
      onSubmit: (event: FormEvent, referrerComponent?: ReferrerComponentType) => {
        if (shouldTriggerCallback(DesignSystemEventProviderAnalyticsEventTypes.OnSubmit, analyticsEvents)) {
          context.callback({
            eventType: DesignSystemEventProviderAnalyticsEventTypes.OnSubmit,
            componentType,
            componentId,
            componentSubType,
            value: undefined,
            shouldStartInteraction: shouldStartInteractionComplete,
            event,
            referrerComponent,
          });
        }
      },
      onValueChange: (value: any) => {
        if (shouldTriggerCallback(DesignSystemEventProviderAnalyticsEventTypes.OnValueChange, analyticsEvents)) {
          context.callback({
            eventType: DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
            componentType,
            componentId,
            componentSubType,
            value: valueHasNoPii ? value : undefined,
            shouldStartInteraction: shouldStartInteractionComplete,
          });
        }
      },
      onView: () => {
        if (shouldTriggerCallback(DesignSystemEventProviderAnalyticsEventTypes.OnView, analyticsEvents)) {
          context.callback({
            eventType: DesignSystemEventProviderAnalyticsEventTypes.OnView,
            componentType,
            componentId,
            componentSubType,
            value: undefined,
            shouldStartInteraction: shouldStartInteractionComplete,
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
export function DesignSystemEventProvider({
  children,
  callback,
}: React.PropsWithChildren<DesignSystemEventProviderContextType>) {
  const contextValue = useMemo(() => {
    return { callback };
  }, [callback]);

  return (
    <DesignSystemEventProviderContext.Provider value={contextValue}>
      {children}
    </DesignSystemEventProviderContext.Provider>
  );
}
