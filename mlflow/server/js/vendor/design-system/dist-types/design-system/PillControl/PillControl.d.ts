import type { RadioGroupProps, RadioGroupItemProps } from '@radix-ui/react-radio-group';
import React from 'react';
import { DesignSystemEventProviderAnalyticsEventTypes } from '../DesignSystemEventProvider';
import type { AnalyticsEventValueChangeNoPiiFlagProps } from '../types';
type RadioGroupSize = 'small' | 'medium' | 'large';
interface RootProps extends Pick<RadioGroupProps, 'defaultValue' | 'value' | 'onValueChange' | 'disabled' | 'name' | 'required' | 'children'>, AnalyticsEventValueChangeNoPiiFlagProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange | DesignSystemEventProviderAnalyticsEventTypes.OnView> {
    size?: RadioGroupSize;
}
export declare const Root: React.ForwardRefExoticComponent<RootProps & React.RefAttributes<HTMLDivElement>>;
interface ItemProps extends Pick<RadioGroupItemProps, 'children' | 'value' | 'disabled' | 'required' | 'onClick'> {
    icon?: React.ReactNode;
    className?: string;
}
export declare const Item: React.ForwardRefExoticComponent<ItemProps & React.RefAttributes<HTMLButtonElement>>;
export {};
//# sourceMappingURL=PillControl.d.ts.map