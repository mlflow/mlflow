import type { RadioGroupProps, RadioGroupItemProps } from '@radix-ui/react-radio-group';
import React from 'react';
import { DesignSystemEventProviderAnalyticsEventTypes } from '../../design-system';
import type { AnalyticsEventValueChangeNoPiiFlagProps } from '../../design-system/types';
type RadioGroupSize = 'small' | 'medium' | 'large';
interface RootProps extends Pick<RadioGroupProps, 'defaultValue' | 'value' | 'onValueChange' | 'disabled' | 'name' | 'required' | 'children'>, AnalyticsEventValueChangeNoPiiFlagProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange> {
    size?: RadioGroupSize;
}
export declare const Root: React.ForwardRefExoticComponent<RootProps & React.RefAttributes<HTMLDivElement>>;
interface ItemProps extends Pick<RadioGroupItemProps, 'children' | 'value' | 'disabled' | 'required'> {
    icon?: React.ReactNode;
}
export declare const Item: React.ForwardRefExoticComponent<ItemProps & React.RefAttributes<HTMLButtonElement>>;
export {};
//# sourceMappingURL=PillControl.d.ts.map