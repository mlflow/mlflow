import {
  useCombobox,
  useMultipleSelection,
  type UseComboboxReturnValue,
  type UseComboboxStateChange,
  type UseMultipleSelectionReturnValue,
} from 'downshift';
import { useMemo } from 'react';

import {
  DesignSystemEventProviderAnalyticsEventTypes,
  DesignSystemEventProviderComponentTypes,
  useDesignSystemEventComponentCallbacks,
} from '../../DesignSystemEventProvider';
import type { AnalyticsEventValueChangeNoPiiFlagProps } from '../../types';

interface SingleSelectProps<T> extends CommonComboboxStateProps<T> {
  multiSelect?: false;
  /* The state setter for inputValue */
  setInputValue?: React.Dispatch<React.SetStateAction<string>>;
  /* The state setter for items */
  setItems: React.Dispatch<React.SetStateAction<T[]>>;
}

interface MultiSelectProps<T> extends CommonComboboxStateProps<T> {
  multiSelect: true;
  /* The state setter for inputValue */
  setInputValue: React.Dispatch<React.SetStateAction<string>>;
  /* The list of currently selected items */
  selectedItems: T[];
  /* The state setter for selectedItems */
  setSelectedItems: React.Dispatch<React.SetStateAction<T[]>>;
}
interface CommonComboboxStateProps<T>
  extends AnalyticsEventValueChangeNoPiiFlagProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange> {
  /* The complete list of items, used for filtering */
  allItems: T[];
  /* The current list of items to render */
  items: T[];
  /* A function that returns whether the item matches the search query */
  matcher?: (item: T, searchQuery: string) => boolean;
  /* A function that returns the string representation of the item. If this arg is not supplied, toString() will be used */
  itemToString?: (item: T) => string;
  /* If using virtualization, this prop should be supplied. */
  onIsOpenChange?: (changes: UseComboboxStateChange<T>) => void;
  /* Whether to allow the input value to not match any options in the list. For this to work, the item type (T) must be a string.
   * Defaults to false.
   */
  allowNewValue?: boolean;
  /* For form libraries like RHF and AntD Form */
  formValue?: T;
  formOnChange?: (value: any) => void;
  formOnBlur?: (value: any) => void;
  onStateChange?: (changes: UseComboboxStateChange<T>) => void;
  /* Sets the initial selected item in the dropdown */
  initialSelectedItem?: T;
  /* Sets the initial input value. Use this prop to set the label for the initial selected item when the component mounts. */
  initialInputValue?: string;
}

const mapItemsToStrings = <T>(items: T[], itemToString?: (item: T) => string): string => {
  return JSON.stringify(items.map((item) => (item ? (itemToString ? itemToString(item) : item.toString()) : '')));
};

export const TypeaheadComboboxStateChangeTypes = useCombobox.stateChangeTypes;
export const TypeaheadComboboxMultiSelectStateChangeTypes = useMultipleSelection.stateChangeTypes;

export type UseComboboxStateProps<T> = SingleSelectProps<T> | MultiSelectProps<T>;
export type ComboboxStateAnalyticsReturnValue<T> = UseComboboxReturnValue<T> &
  AnalyticsEventValueChangeNoPiiFlagProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange>;

export function useComboboxState<T>({
  allItems,
  items,
  itemToString,
  onIsOpenChange,
  allowNewValue = false,
  formValue,
  formOnChange,
  formOnBlur,
  componentId,
  valueHasNoPii,
  analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange],
  ...props
}: UseComboboxStateProps<T>): ComboboxStateAnalyticsReturnValue<T> {
  function getFilteredItems(inputValue: string | undefined) {
    const lowerCasedInputValue = inputValue?.toLowerCase() ?? '';
    // If the input is empty or if there is no matcher supplied, do not filter
    return allItems.filter((item: T) => !inputValue || !props.matcher || props.matcher(item, lowerCasedInputValue));
  }

  const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.TypeaheadCombobox,
    componentId,
    analyticsEvents: memoizedAnalyticsEvents,
    valueHasNoPii,
  });

  const comboboxState = {
    ...useCombobox({
      onIsOpenChange: onIsOpenChange,
      onInputValueChange: ({ inputValue }) => {
        if (inputValue !== undefined) {
          props.setInputValue?.(inputValue);
        }
        if (!props.multiSelect) {
          props.setItems(getFilteredItems(inputValue));
        }
      },
      items: items,
      itemToString(item) {
        return item ? (itemToString ? itemToString(item) : item.toString()) : '';
      },
      defaultHighlightedIndex: props.multiSelect ? 0 : undefined, // after selection for multiselect, highlight the first item.
      scrollIntoView: () => {}, // disabling scroll because floating-ui repositions the menu
      selectedItem: props.multiSelect ? null : formValue, // useMultipleSelection will handle the item selection for multiselect
      stateReducer(state, actionAndChanges) {
        const { changes, type } = actionAndChanges;

        switch (type) {
          case useCombobox.stateChangeTypes.InputBlur:
            if (!props.multiSelect) {
              // If allowNewValue is true, register the input's current value on blur
              if (allowNewValue) {
                const newInputValue = state.inputValue === '' ? null : state.inputValue;
                formOnChange?.(newInputValue);
                formOnBlur?.(newInputValue);
              } else {
                // If allowNewValue is false, clear value on blur
                formOnChange?.(null);
                formOnBlur?.(null);
              }
            } else {
              formOnBlur?.(state.selectedItem);
            }
            return changes;
          case useCombobox.stateChangeTypes.InputKeyDownEnter:
          case useCombobox.stateChangeTypes.ItemClick:
            formOnChange?.(changes.selectedItem);
            return {
              ...changes,
              highlightedIndex: props.multiSelect ? state.highlightedIndex : 0, // on multiselect keep the highlighted index unchanged.
              isOpen: props.multiSelect ? true : false, // for multiselect, keep the menu open after selection.
            };
          default:
            return changes;
        }
      },
      onStateChange: (args) => {
        const { type, selectedItem: newSelectedItem, inputValue: newInputValue } = args;

        props.onStateChange?.(args);

        if (props.multiSelect) {
          switch (type) {
            case useCombobox.stateChangeTypes.InputKeyDownEnter:
            case useCombobox.stateChangeTypes.ItemClick:
            case useCombobox.stateChangeTypes.InputBlur:
              if (newSelectedItem) {
                props.setSelectedItems([...props.selectedItems, newSelectedItem]);
                props.setInputValue('');
                formOnBlur?.([...props.selectedItems, newSelectedItem]);
              }
              break;
            case useCombobox.stateChangeTypes.InputChange:
              props.setInputValue(newInputValue ?? '');
              break;
            case useCombobox.stateChangeTypes.FunctionReset:
              eventContext.onValueChange?.('[]');
              break;
            default:
              break;
          }
          // Unselect when clicking selected item
          if (newSelectedItem && props.selectedItems.includes(newSelectedItem)) {
            const newSelectedItems = props.selectedItems.filter((item) => item !== newSelectedItem);
            props.setSelectedItems(newSelectedItems);
            eventContext.onValueChange?.(mapItemsToStrings(newSelectedItems, itemToString));
          } else if (newSelectedItem) {
            eventContext.onValueChange?.(mapItemsToStrings([...props.selectedItems, newSelectedItem], itemToString));
          }
        } else if (newSelectedItem) {
          eventContext.onValueChange?.(itemToString ? itemToString(newSelectedItem) : newSelectedItem.toString());
        } else if (type === useCombobox.stateChangeTypes.FunctionReset) {
          eventContext.onValueChange?.('');
        }
      },
      initialInputValue: props.initialInputValue,
      initialSelectedItem: props.initialSelectedItem,
    }),
    componentId,
    analyticsEvents,
    valueHasNoPii,
  };

  return comboboxState;
}

interface AnalyticsConfig
  extends AnalyticsEventValueChangeNoPiiFlagProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange> {
  itemToString?: (item: any) => string;
}

export function useMultipleSelectionState<T>(
  selectedItems: T[],
  setSelectedItems: React.Dispatch<React.SetStateAction<T[]>>,
  {
    componentId,
    analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange],
    valueHasNoPii,
    itemToString,
  }: AnalyticsConfig,
): UseMultipleSelectionReturnValue<T> {
  const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.TypeaheadCombobox,
    componentId,
    analyticsEvents: memoizedAnalyticsEvents,
    valueHasNoPii,
  });

  return useMultipleSelection({
    selectedItems,
    onStateChange({ selectedItems: newSelectedItems, type }) {
      switch (type) {
        case useMultipleSelection.stateChangeTypes.SelectedItemKeyDownBackspace:
        case useMultipleSelection.stateChangeTypes.SelectedItemKeyDownDelete:
        case useMultipleSelection.stateChangeTypes.DropdownKeyDownBackspace:
        case useMultipleSelection.stateChangeTypes.FunctionRemoveSelectedItem:
          setSelectedItems(newSelectedItems || []);
          break;
        default:
          break;
      }
      eventContext.onValueChange?.(mapItemsToStrings(newSelectedItems || [], itemToString));
    },
  });
}
