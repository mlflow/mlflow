import type { AlertProps as AntDAlertProps } from 'antd';
import { DesignSystemEventProviderAnalyticsEventTypes } from '../DesignSystemEventProvider';
import type { AnalyticsEventProps, DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
export type AlertType = NonNullable<Exclude<AntDAlertProps['type'], 'success'>>;
export interface AlertProps extends Omit<AntDAlertProps, 'closeText' | 'showIcon' | 'type' | 'icon'>, HTMLDataAttributes, DangerouslySetAntdProps<AntDAlertProps>, AnalyticsEventProps<DesignSystemEventProviderAnalyticsEventTypes.OnView> {
    type: AlertType;
    closeIconLabel?: string;
    /**
     * @deprecated Use CEP-governed banners instead. go/banner/create go/getabanner go/banner
     */
    banner?: boolean;
}
export declare const Alert: React.FC<AlertProps>;
//# sourceMappingURL=Alert.d.ts.map