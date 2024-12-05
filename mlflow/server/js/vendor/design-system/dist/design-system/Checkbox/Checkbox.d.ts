import type { CheckboxProps as AntDCheckboxProps, CheckboxGroupProps as AntDCheckboxGroupProps, CheckboxChangeEvent } from 'antd/lib/checkbox';
import type { CheckboxValueType as AntDCheckboxValueType } from 'antd/lib/checkbox/Group';
import type { Theme } from '../../theme';
import { DesignSystemEventProviderAnalyticsEventTypes } from '../DesignSystemEventProvider';
import type { AnalyticsEventProps, DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
export type CheckboxValueType = AntDCheckboxValueType;
export declare const getWrapperStyle: ({ clsPrefix, theme, wrapperStyle, }: {
    clsPrefix: string;
    theme: Theme;
    wrapperStyle?: React.CSSProperties;
    useNewStyles?: boolean;
}) => import("@emotion/utils").SerializedStyles;
export interface CheckboxProps extends DangerouslySetAntdProps<AntDCheckboxProps>, Omit<React.InputHTMLAttributes<HTMLInputElement>, 'onChange' | 'checked'>, HTMLDataAttributes, AnalyticsEventProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange> {
    isChecked?: boolean | null;
    onChange?: (isChecked: boolean, event: CheckboxChangeEvent) => void;
    children?: React.ReactNode;
    isDisabled?: boolean;
    /**
     * Used to set styling for the div wrapping the checkbox element
     */
    wrapperStyle?: React.CSSProperties;
    /**
     * Used to style the checkbox element itself
     */
    style?: React.CSSProperties;
}
export interface CheckboxGroupProps extends Omit<AntDCheckboxGroupProps, 'prefixCls'>, Omit<React.InputHTMLAttributes<HTMLInputElement>, 'defaultValue' | 'onChange' | 'value'> {
    layout?: 'vertical' | 'horizontal';
}
export declare const Checkbox: import("react").ForwardRefExoticComponent<CheckboxProps & import("react").RefAttributes<HTMLInputElement>> & {
    Group: import("react").ForwardRefExoticComponent<CheckboxGroupProps & import("react").RefAttributes<HTMLInputElement>>;
};
export declare const __INTERNAL_DO_NOT_USE__Group: import("react").ForwardRefExoticComponent<CheckboxGroupProps & import("react").RefAttributes<HTMLInputElement>>;
//# sourceMappingURL=Checkbox.d.ts.map