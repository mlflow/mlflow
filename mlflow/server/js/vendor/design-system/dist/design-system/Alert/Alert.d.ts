/// <reference types="react" />
import type { AlertProps as AntDAlertProps } from 'antd';
import type { DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
type AlertType = NonNullable<Exclude<AntDAlertProps['type'], 'success'>>;
export interface AlertProps extends Omit<AntDAlertProps, 'closeText' | 'showIcon' | 'type' | 'icon'>, HTMLDataAttributes, DangerouslySetAntdProps<AntDAlertProps> {
    type: AlertType;
}
export declare const Alert: React.FC<AlertProps>;
export {};
//# sourceMappingURL=Alert.d.ts.map