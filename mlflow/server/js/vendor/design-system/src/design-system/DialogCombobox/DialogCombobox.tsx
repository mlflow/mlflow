import * as Popover from '@radix-ui/react-popover';
import type { ReactNode } from 'react';
import { useState, useEffect, useCallback, useMemo } from 'react';

import { useDialogComboboxContext } from './hooks/useDialogComboboxContext';
import { DialogComboboxContextProvider } from './providers/DialogComboboxContext';
import {
  ComponentFinderContext,
  DesignSystemEventProviderAnalyticsEventTypes,
  DesignSystemEventProviderComponentTypes,
  useDesignSystemEventComponentCallbacks,
} from '../DesignSystemEventProvider';
import type { AnalyticsEventValueChangeNoPiiFlagProps, HTMLDataAttributes } from '../types';

export type ConditionalOptionalLabel = { id?: string; label: ReactNode } | { id: string; label?: ReactNode };

export interface DialogComboboxProps
  extends Popover.PopoverProps,
    HTMLDataAttributes,
    AnalyticsEventValueChangeNoPiiFlagProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange> {
  value?: string[];
  stayOpenOnSelection?: boolean;
  multiSelect?: boolean;
  emptyText?: string;
  scrollToSelectedElement?: boolean;
  rememberLastScrollPosition?: boolean;
}

export const DialogCombobox = ({
  children,
  label,
  id,
  value = [],
  open,
  emptyText,
  scrollToSelectedElement = true,
  rememberLastScrollPosition = false,
  componentId,
  analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange],
  valueHasNoPii,
  ...props
}: DialogComboboxProps & ConditionalOptionalLabel) => {
  // Used to avoid infinite loop when value is controlled from within the component (DialogComboboxOptionControlledList)
  // We can't remove setValue altogether because uncontrolled component users need to be able to set the value from root for trigger to update
  const [isControlled, setIsControlled] = useState(false);
  const [selectedValue, setSelectedValue] = useState<string[]>(value);
  const [isOpen, setIsOpen] = useState<boolean>(Boolean(open));
  const [contentWidth, setContentWidth] = useState<number | string | undefined>();
  const [textOverflowMode, setTextOverflowMode] = useState<'ellipsis' | 'multiline'>('multiline');
  const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.DialogCombobox,
    componentId,
    analyticsEvents: memoizedAnalyticsEvents,
    valueHasNoPii,
  });

  const setSelectedValueWrapper = useCallback(
    (newValue: string[]) => {
      eventContext.onValueChange(JSON.stringify(newValue));
      setSelectedValue(newValue);
    },
    [eventContext],
  );

  useEffect(() => {
    if (
      ((!Array.isArray(selectedValue) || !Array.isArray(value)) && selectedValue !== value) ||
      (selectedValue && value && selectedValue.length === value.length && selectedValue.every((v, i) => v === value[i]))
    ) {
      return;
    }

    if (!isControlled) {
      setSelectedValueWrapper(value);
    }
  }, [value, isControlled, selectedValue, setSelectedValueWrapper]);

  return (
    <DialogComboboxContextProvider
      value={{
        id,
        label,
        value: selectedValue,
        setValue: setSelectedValueWrapper,
        setIsControlled,
        contentWidth,
        setContentWidth,
        textOverflowMode,
        setTextOverflowMode,
        isInsideDialogCombobox: true,
        multiSelect: props.multiSelect,
        stayOpenOnSelection: props.stayOpenOnSelection,
        isOpen,
        setIsOpen,
        emptyText,
        scrollToSelectedElement,
        rememberLastScrollPosition,
        componentId,
        analyticsEvents,
        valueHasNoPii,
      }}
    >
      <Root open={open !== undefined ? open : isOpen} {...props}>
        <ComponentFinderContext.Provider value={{ dataComponentProps: eventContext.dataComponentProps }}>
          {children}
        </ComponentFinderContext.Provider>
      </Root>
    </DialogComboboxContextProvider>
  );
};

const Root = (props: Partial<DialogComboboxProps>) => {
  const { children, stayOpenOnSelection, multiSelect, onOpenChange, ...restProps } = props;
  const { value, setIsOpen } = useDialogComboboxContext();

  const handleOpenChange = (open: boolean) => {
    setIsOpen(open);
    onOpenChange?.(open);
  };

  useEffect(() => {
    if (!stayOpenOnSelection && (typeof stayOpenOnSelection === 'boolean' || !multiSelect)) {
      setIsOpen(false);
    }
  }, [value, stayOpenOnSelection, multiSelect, setIsOpen]);

  return (
    <Popover.Root onOpenChange={handleOpenChange} {...restProps}>
      {children}
    </Popover.Root>
  );
};
