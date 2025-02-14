import React, {
  createContext,
  forwardRef,
  useContext,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from 'react';

import type { SelectContentProps, SelectOptionGroupProps, SelectOptionProps, SelectTriggerProps } from '.';
import { Select, SelectContent, SelectOption, SelectOptionGroup, SelectTrigger } from '.';
import {
  DesignSystemEventProviderAnalyticsEventTypes,
  DesignSystemEventProviderComponentTypes,
  useDesignSystemEventComponentCallbacks,
} from '../DesignSystemEventProvider';
import type { ConditionalOptionalLabel } from '../DialogCombobox';
import { useDialogComboboxContext } from '../DialogCombobox/hooks/useDialogComboboxContext';
import type { AnalyticsEventValueChangeNoPiiFlagProps } from '../types';

interface SimpleSelectContextValue {
  value?: string;
  onChange: (value: string) => void;
}

const SimpleSelectContext = createContext<SimpleSelectContextValue | undefined>(undefined);

// This is modelled to match the behavior of native select elements;
// this allows SimpleSelect to be compatible natively with React Hook Form.
export interface SimpleSelectChangeEventType {
  target: {
    name?: string;
    type: string;
    value: string;
  };
  type: string;
}

export interface SimpleSelectProps
  extends Omit<SelectTriggerProps, 'onChange' | 'value' | 'defaultValue' | 'onClear' | 'label'>,
    AnalyticsEventValueChangeNoPiiFlagProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange> {
  /** For an uncontrolled `SimpleSelect`, optionally specify a defaultValue. Will be ignored if using `value` to control state of the component. */
  defaultValue?: string;
  value?: string;
  name?: string;
  validationState?: SelectTriggerProps['validationState'];
  contentProps?: SelectContentProps;
  onChange?: (e: SimpleSelectChangeEventType) => void;
  forceCloseOnEscape?: boolean;
  /** Callback returning the open state of the dropdown. */
  onOpenChange?: (isOpen: boolean) => void;
}

const getSelectedOption = (children: React.ReactNode, value: string): React.ReactNode | undefined => {
  const childArray = React.Children.toArray(children);
  for (const child of childArray) {
    if (React.isValidElement(child)) {
      if (child.type === SimpleSelectOption && child.props.value === value) {
        return child;
      }
      if (child.props.children) {
        const nestedOption = getSelectedOption(child.props.children, value);
        if (nestedOption) {
          return nestedOption;
        }
      }
    }
  }
  return undefined;
};

const getSelectedOptionLabel = (children: React.ReactNode, value: string): string => {
  const selectedOption = getSelectedOption(children, value);
  if (React.isValidElement(selectedOption)) {
    return selectedOption.props.children;
  }
  return '';
};

/**
 * This is the future `Select` component which simplifies the API of the current Select primitives.
 * It is temporarily named `SimpleSelect` pending cleanup.
 */
export const SimpleSelect = forwardRef<HTMLInputElement, SimpleSelectProps & ConditionalOptionalLabel>(
  (
    {
      defaultValue,
      name,
      placeholder,
      children,
      contentProps,
      onChange,
      onOpenChange,
      id,
      label,
      value,
      validationState,
      forceCloseOnEscape,
      componentId,
      analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange],
      valueHasNoPii,
      ...rest
    }: SimpleSelectProps & ConditionalOptionalLabel,
    ref,
  ) => {
    const [defaultLabel] = useState<string>(() => {
      if (value) {
        return getSelectedOptionLabel(children, value);
      }
      return '';
    });

    const innerRef = useRef<HTMLInputElement>(null);
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion -- TODO(FEINF-3982)
    useImperativeHandle(ref, () => innerRef.current!, []);
    const previousExternalValue = useRef(value);
    const [internalValue, setInternalValue] = useState(value);
    const [selectedLabel, setSelectedLabel] = useState(defaultLabel);
    const isControlled = value !== undefined;

    const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
    const eventContext = useDesignSystemEventComponentCallbacks({
      componentType: DesignSystemEventProviderComponentTypes.SimpleSelect,
      componentId,
      analyticsEvents: memoizedAnalyticsEvents,
      valueHasNoPii,
    });

    // Controlled state setup.
    useEffect(() => {
      if (value !== undefined && value !== previousExternalValue.current) {
        setInternalValue(value);
        previousExternalValue.current = value;
      }
    }, [value]);

    // Uncontrolled state setup.
    useEffect(() => {
      if (isControlled) {
        return;
      }

      // Set initial state.
      const element = innerRef.current;
      const initialValue = defaultValue || element?.value || '';
      setInternalValue(initialValue);
      previousExternalValue.current = initialValue;
    }, [isControlled, defaultValue, value]);

    // Separately update the label when the value changes; this responds
    // to either the controlled or uncontrolled useEffects above.
    useEffect(() => {
      setSelectedLabel(getSelectedOptionLabel(children, internalValue || ''));
    }, [internalValue, children]);

    // Handles controlled state, and propagates changes to the input element.
    const handleChange = (newValue: string) => {
      eventContext.onValueChange(newValue);
      innerRef.current?.setAttribute('value', newValue || '');
      setInternalValue(newValue);
      setSelectedLabel(getSelectedOptionLabel(children, newValue));

      if (onChange) {
        onChange({
          target: {
            name,
            type: 'select',
            value: newValue,
          },
          type: 'change',
        });
      }
    };

    return (
      <SimpleSelectContext.Provider value={{ value: internalValue, onChange: handleChange }}>
        <Select
          // SimpleSelect emits its own value change events rather than emitting them from the underlying
          // DialogCombobox due to how SimpleSelect sets its initial state. The Select componentId is explicitly
          // set to undefined to prevent it from emitting events if the componentId prop is required in the future.
          componentId={undefined as unknown as string}
          value={internalValue}
          placeholder={placeholder}
          label={label ?? rest['aria-label']}
          id={id}
        >
          <SimpleSelectContentWrapper onOpenChange={onOpenChange}>
            <SelectTrigger
              {...rest}
              validationState={validationState}
              onClear={() => {
                handleChange('');
              }}
              id={id}
              value={internalValue}
              {...eventContext.dataComponentProps}
            >
              {selectedLabel || placeholder}
            </SelectTrigger>
            {/* TODO: Trigger is a Button, which does not have the onChange events required by RHF; we should look at improving this.
             *  For now, we add a hidden input to act as the ref / external input to ensure RHF works as expected.
             */}
            <input type="hidden" ref={innerRef} />
            <SelectContent forceCloseOnEscape={forceCloseOnEscape} {...contentProps}>
              {children}
            </SelectContent>
          </SimpleSelectContentWrapper>
        </Select>
      </SimpleSelectContext.Provider>
    );
  },
);

interface SimpleSelectContentWrapperProps {
  onOpenChange?: (isOpen: boolean) => void;
}

// This component is used to propagate the open state of the DialogCombobox to the SimpleSelect.
// We don't directly pass through `onOpenChange` since it's tied into the actual state; `SimpleSelect` merely
// needs to communicate via the optional prop if the dropdown is open or not and doesn't need to control it.
const SimpleSelectContentWrapper: React.FC<SimpleSelectContentWrapperProps> = ({ children, onOpenChange }) => {
  const { isOpen } = useDialogComboboxContext();

  useEffect(() => {
    if (onOpenChange) {
      onOpenChange(Boolean(isOpen));
    }
  }, [isOpen, onOpenChange]);

  return <>{children}</>;
};

export type SimpleSelectOptionProps = Omit<SelectOptionProps, 'hintColumn' | 'hintColumnWidthPercent'>;

export const SimpleSelectOption = forwardRef<HTMLDivElement, SimpleSelectOptionProps>(
  ({ value, children, ...rest }, ref) => {
    const context = useContext(SimpleSelectContext);

    if (!context) {
      throw new Error('SimpleSelectOption must be used within a SimpleSelect');
    }

    const { onChange } = context;

    return (
      <SelectOption
        {...rest}
        ref={ref}
        value={value}
        onChange={({ value }) => {
          onChange(value);
        }}
      >
        {children}
      </SelectOption>
    );
  },
);

export interface SimpleSelectOptionGroupProps extends Omit<SelectOptionGroupProps, 'name'> {
  label: string;
}

export const SimpleSelectOptionGroup = ({ children, label, ...props }: SimpleSelectOptionGroupProps) => {
  const context = useContext(SimpleSelectContext);

  if (!context) {
    throw new Error('SimpleSelectOptionGroup must be used within a SimpleSelect');
  }

  return (
    <SelectOptionGroup {...props} name={label}>
      {children}
    </SelectOptionGroup>
  );
};
