import type { SerializedStyles } from '@emotion/react';
import type { RadioGroupProps as AntDRadioGroupProps, RadioProps as AntDRadioProps, RadioChangeEvent } from 'antd';
import React from 'react';
import type { Theme } from '../../theme';
import { DesignSystemEventProviderAnalyticsEventTypes } from '../DesignSystemEventProvider/DesignSystemEventProvider';
import type { AnalyticsEventValueChangeNoPiiFlagOptionalProps, AnalyticsEventValueChangeNoPiiFlagProps, DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
export type { RadioChangeEvent } from 'antd';
export declare const useRadioGroupContext: () => {
    componentId: string;
    value: string;
    onChange: (event: RadioChangeEvent) => void;
};
export declare const getRadioStyles: ({ theme, clsPrefix, useNewShadows, }: {
    theme: Theme;
    clsPrefix: string;
    useNewShadows: boolean;
}) => SerializedStyles;
export interface RadioProps extends Omit<AntDRadioProps, 'prefixCls' | 'type' | 'skipGroup'>, DangerouslySetAntdProps<AntDRadioGroupProps>, HTMLDataAttributes, AnalyticsEventValueChangeNoPiiFlagOptionalProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange> {
    __INTERNAL_DISABLE_RADIO_ROLE?: boolean;
}
export interface RadioGroupProps extends Omit<AntDRadioGroupProps, 'optionType' | 'buttonStyle' | 'size' | 'prefixCls' | 'skipGroup'>, DangerouslySetAntdProps<AntDRadioGroupProps>, HTMLDataAttributes, AnalyticsEventValueChangeNoPiiFlagProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange> {
    layout?: 'vertical' | 'horizontal';
    name: string;
    useEqualColumnWidths?: boolean;
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
export declare const __INTERNAL_DO_NOT_USE__VerticalGroup: React.FC<RadioGroupProps>;
export declare const __INTERNAL_DO_NOT_USE__HorizontalGroup: React.FC<OrientedRadioGroupProps>;
//# sourceMappingURL=Radio.d.ts.map