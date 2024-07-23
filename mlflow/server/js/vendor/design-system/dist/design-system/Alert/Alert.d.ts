import type { AlertProps as AntDAlertProps } from 'antd';
import { DesignSystemEventProviderAnalyticsEventTypes } from '../DesignSystemEventProvider';
import type { AnalyticsEventOptionalProps, DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
type AlertType = NonNullable<Exclude<AntDAlertProps['type'], 'success'>>;
export interface AlertProps extends Omit<AntDAlertProps, 'closeText' | 'showIcon' | 'type' | 'icon'>, HTMLDataAttributes, DangerouslySetAntdProps<AntDAlertProps>, AnalyticsEventOptionalProps<DesignSystemEventProviderAnalyticsEventTypes.OnView> {
    type: AlertType;
    closeIconLabel?: string;
}
export declare const Alert: React.FC<AlertProps>;
export {};
//# sourceMappingURL=Alert.d.ts.map