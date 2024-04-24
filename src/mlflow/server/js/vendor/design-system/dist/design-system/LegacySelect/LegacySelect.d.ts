import type { SelectProps as AntDSelectProps } from 'antd';
import { Select as AntDSelect } from 'antd';
import type { RefSelectProps as AntdRefSelectProps, SelectValue as AntdSelectValue } from 'antd/lib/select';
import React from 'react';
import type { WithLoadingState } from '../LoadingState/LoadingState';
import type { DangerouslySetAntdProps, FormElementValidationState, HTMLDataAttributes } from '../types';
export type SelectValue = AntdSelectValue;
type SelectRef = React.Ref<AntdRefSelectProps>;
type OmittedProps = 'bordered' | 'autoClearSearchValue' | 'dropdownRender' | 'dropdownStyle' | 'size' | 'suffixIcon' | 'tagRender' | 'clearIcon' | 'removeIcon' | 'showArrow' | 'dropdownMatchSelectWidth' | 'menuItemSelectedIcon' | 'showSearch';
export interface SelectProps<T = string> extends Omit<AntDSelectProps<T>, OmittedProps>, FormElementValidationState, HTMLDataAttributes, DangerouslySetAntdProps<Pick<AntDSelectProps<T>, OmittedProps>>, Omit<WithLoadingState, 'loading'> {
    maxHeight?: number;
}
export interface SelectOptionProps extends DangerouslySetAntdProps<typeof AntDSelect.Option> {
    value: string | number;
    disabled?: boolean;
    key?: string | number;
    title?: string;
    label?: React.ReactNode;
    children: React.ReactNode;
    'data-testid'?: string;
    onClick?: () => void;
    className?: string;
    style?: React.CSSProperties;
}
/** @deprecated Use `SelectOptionProps` */
export interface OptionProps extends SelectOptionProps {
}
export declare const SelectOption: React.ForwardRefExoticComponent<SelectOptionProps & React.RefAttributes<HTMLElement>> & {
    isSelectOption: boolean;
};
/**
 * @deprecated use Select.Option instead
 */
export declare const Option: React.ForwardRefExoticComponent<SelectOptionProps & React.RefAttributes<HTMLElement>> & {
    isSelectOption: boolean;
};
export interface SelectOptGroupProps {
    key?: string | number;
    label: React.ReactNode;
    className?: string;
    style?: React.CSSProperties;
    children?: React.ReactNode;
}
/** @deprecated Use `SelectOptGroupProps` */
export interface OptGroupProps extends SelectOptGroupProps {
}
export declare const SelectOptGroup: React.ForwardRefExoticComponent<SelectOptGroupProps & React.RefAttributes<HTMLElement>> & {
    isSelectOptGroup: boolean;
};
/**
 * @deprecated use Select.OptGroup instead
 */
export declare const OptGroup: React.ForwardRefExoticComponent<SelectOptGroupProps & React.RefAttributes<HTMLElement>> & {
    isSelectOptGroup: boolean;
};
/**
 * @deprecated Use SelectV2, TypeaheadCombobox, or DialogCombobox depending on your use-case. See http://go/deprecate-ant-select for more information
 */
export declare const LegacySelect: (<T extends AntdSelectValue>(props: SelectProps<T> & {
    ref?: SelectRef | undefined;
}) => JSX.Element) & {
    Option: typeof SelectOption;
    OptGroup: typeof SelectOptGroup;
} & React.ForwardRefExoticComponent<SelectProps<string> & React.RefAttributes<HTMLElement>>;
export {};
//# sourceMappingURL=LegacySelect.d.ts.map