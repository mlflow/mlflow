import type { SerializedStyles } from '@emotion/react';
import type { RadioGroupProps as AntDRadioGroupProps, RadioProps as AntDRadioProps } from 'antd';
import type { Theme } from '../../theme';
import { DesignSystemEventProviderAnalyticsEventTypes } from '../DesignSystemEventProvider/DesignSystemEventProvider';
import type { AnalyticsEventValueChangeNoPiiFlagProps, DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
export type { RadioChangeEvent } from 'antd';
export declare const getRadioStyles: ({ theme, clsPrefix }: {
    theme: Theme;
    clsPrefix: string;
}) => SerializedStyles;
export interface RadioProps extends Omit<AntDRadioProps, 'prefixCls' | 'type' | 'skipGroup'>, DangerouslySetAntdProps<AntDRadioGroupProps>, HTMLDataAttributes {
}
export interface RadioGroupProps extends Omit<AntDRadioGroupProps, 'optionType' | 'buttonStyle' | 'size' | 'prefixCls' | 'skipGroup'>, DangerouslySetAntdProps<AntDRadioGroupProps>, HTMLDataAttributes, AnalyticsEventValueChangeNoPiiFlagProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange> {
    layout?: 'vertical' | 'horizontal';
    name: string;
}
interface OrientedRadioGroupProps extends Omit<RadioGroupProps, 'layout'> {
}
export interface RadioInterface extends React.FC<RadioProps> {
    Group: typeof Group;
    HorizontalGroup: typeof HorizontalGroup;
}
declare const HorizontalGroup: React.FC<OrientedRadioGroupProps>;
declare const Group: React.FC<RadioGroupProps>;
export declare const Radio: RadioInterface;
export declare const __INTERNAL_DO_NOT_USE__VerticalGroup: import("react").FC<RadioGroupProps>;
export declare const __INTERNAL_DO_NOT_USE__HorizontalGroup: import("react").FC<OrientedRadioGroupProps>;
//# sourceMappingURL=Radio.d.ts.map