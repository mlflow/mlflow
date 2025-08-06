import type { SelectProps as AntDSelectProps } from 'antd';
import { Select as AntDSelect } from 'antd';
import type { RefSelectProps as AntdRefSelectProps, SelectValue as AntdSelectValue } from 'antd/lib/select';
import React from 'react';
import type { WithLoadingState } from '../LoadingState/LoadingState';
import type { DangerouslySetAntdProps, FormElementValidationState, HTMLDataAttributes } from '../types';
export type LegacySelectValue = AntdSelectValue;
type SelectRef = React.Ref<AntdRefSelectProps>;
type OmittedProps = 'bordered' | 'autoClearSearchValue' | 'dropdownRender' | 'dropdownStyle' | 'size' | 'suffixIcon' | 'tagRender' | 'clearIcon' | 'removeIcon' | 'showArrow' | 'dropdownMatchSelectWidth' | 'menuItemSelectedIcon' | 'showSearch';
export interface LegacySelectProps<T = string> extends Omit<AntDSelectProps<T>, OmittedProps>, FormElementValidationState, HTMLDataAttributes, DangerouslySetAntdProps<Pick<AntDSelectProps<T>, OmittedProps>>, Omit<WithLoadingState, 'loading'> {
    maxHeight?: number;
}
export interface LegacySelectOptionProps extends DangerouslySetAntdProps<typeof AntDSelect.Option> {
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
/** @deprecated Use `LegacySelectOptionProps` */
export interface LegacyOptionProps extends LegacySelectOptionProps {
}
export declare const LegacySelectOption: React.ForwardRefExoticComponent<LegacySelectOptionProps & React.RefAttributes<HTMLElement>> & {
    isSelectOption: boolean;
};
/**
 * @deprecated use LegacySelect.Option instead
 */
export declare const LegacyOption: React.ForwardRefExoticComponent<LegacySelectOptionProps & React.RefAttributes<HTMLElement>> & {
    isSelectOption: boolean;
};
export interface LegacySelectOptGroupProps {
    key?: string | number;
    label: React.ReactNode;
    className?: string;
    style?: React.CSSProperties;
    children?: React.ReactNode;
}
/** @deprecated Use `LegacySelectOptGroupProps` */
export interface LegacyOptGroupProps extends LegacySelectOptGroupProps {
}
export declare const LegacySelectOptGroup: React.ForwardRefExoticComponent<LegacySelectOptGroupProps & React.RefAttributes<HTMLElement>> & {
    isSelectOptGroup: boolean;
};
/**
 * @deprecated use LegacySelect.OptGroup instead
 */
export declare const LegacyOptGroup: React.ForwardRefExoticComponent<LegacySelectOptGroupProps & React.RefAttributes<HTMLElement>> & {
    isSelectOptGroup: boolean;
};
/**
 * @deprecated Use Select, TypeaheadCombobox, or DialogCombobox depending on your use-case. See http://go/deprecate-ant-select for more information
 */
export declare const LegacySelect: (<T extends LegacySelectValue>(props: LegacySelectProps<T> & {
    ref?: SelectRef;
}) => JSX.Element) & {
    Option: typeof LegacySelectOption;
    OptGroup: typeof LegacySelectOptGroup;
} & React.ForwardRefExoticComponent<LegacySelectProps<string> & React.RefAttributes<HTMLElement>>;
export {};
//# sourceMappingURL=LegacySelect.d.ts.map