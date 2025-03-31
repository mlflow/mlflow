import React from 'react';
import type { SelectContentProps, SelectOptionGroupProps, SelectOptionProps, SelectTriggerProps } from '.';
import { DesignSystemEventProviderAnalyticsEventTypes } from '../DesignSystemEventProvider';
import type { ConditionalOptionalLabel } from '../DialogCombobox';
import type { AnalyticsEventValueChangeNoPiiFlagProps } from '../types';
export interface SimpleSelectChangeEventType {
    target: {
        name?: string;
        type: string;
        value: string;
    };
    type: string;
}
export interface SimpleSelectProps extends Omit<SelectTriggerProps, 'onChange' | 'value' | 'defaultValue' | 'onClear' | 'label'>, AnalyticsEventValueChangeNoPiiFlagProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange> {
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
/**
 * This is the future `Select` component which simplifies the API of the current Select primitives.
 * It is temporarily named `SimpleSelect` pending cleanup.
 */
export declare const SimpleSelect: React.ForwardRefExoticComponent<(SimpleSelectProps & ConditionalOptionalLabel) & React.RefAttributes<HTMLInputElement>>;
export type SimpleSelectOptionProps = Omit<SelectOptionProps, 'hintColumn' | 'hintColumnWidthPercent'>;
export declare const SimpleSelectOption: React.ForwardRefExoticComponent<SimpleSelectOptionProps & React.RefAttributes<HTMLDivElement>>;
export interface SimpleSelectOptionGroupProps extends Omit<SelectOptionGroupProps, 'name'> {
    label: string;
}
export declare const SimpleSelectOptionGroup: ({ children, label, ...props }: SimpleSelectOptionGroupProps) => import("@emotion/react/jsx-runtime").JSX.Element;
//# sourceMappingURL=SimpleSelect.d.ts.map