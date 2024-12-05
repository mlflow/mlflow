import type { WithConditionalCSSProp } from '@emotion/react/types/jsx-namespace';
import type { SwitchProps as AntDSwitchProps } from 'antd';
import { DesignSystemEventProviderAnalyticsEventTypes } from '../DesignSystemEventProvider';
import type { LabelProps } from '../Label/Label';
import type { AnalyticsEventProps, DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
export interface SwitchProps extends Pick<AntDSwitchProps, 'autoFocus' | 'checked' | 'checkedChildren' | 'className' | 'defaultChecked' | 'disabled' | 'unCheckedChildren' | 'onChange' | 'onClick'>, HTMLDataAttributes, DangerouslySetAntdProps<AntDSwitchProps>, AnalyticsEventProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange> {
    id?: string;
    /**
     * Label for the Switch, provided as prop for styling purposes
     */
    label?: string;
    labelProps?: LabelProps & WithConditionalCSSProp<LabelProps>;
    activeLabel?: React.ReactNode;
    inactiveLabel?: React.ReactNode;
    disabledLabel?: React.ReactNode;
}
export declare const Switch: React.FC<SwitchProps>;
//# sourceMappingURL=Switch.d.ts.map