/// <reference types="react" />
import type { SerializedStyles } from '@emotion/react';
import type { RadioGroupProps as AntDRadioGroupProps, RadioProps as AntDRadioProps } from 'antd';
import type { Theme } from '../../theme';
import { DesignSystemEventProviderAnalyticsEventTypes } from '../DesignSystemEventProvider/DesignSystemEventProvider';
import type { AnalyticsEventValueChangeNoPiiFlagOptionalProps, DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
export type { RadioChangeEvent } from 'antd';
export declare const getRadioStyles: ({ theme, clsPrefix, useNewStyles, }: {
    theme: Theme;
    clsPrefix: string;
    useNewStyles?: boolean | undefined;
}) => SerializedStyles;
export interface RadioProps extends Omit<AntDRadioProps, 'prefixCls' | 'type' | 'skipGroup'>, DangerouslySetAntdProps<AntDRadioGroupProps>, HTMLDataAttributes {
}
export interface RadioGroupProps extends Omit<AntDRadioGroupProps, 'optionType' | 'buttonStyle' | 'size' | 'prefixCls' | 'skipGroup'>, DangerouslySetAntdProps<AntDRadioGroupProps>, HTMLDataAttributes, AnalyticsEventValueChangeNoPiiFlagOptionalProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange> {
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