import type { AlertProps as AntDAlertProps } from 'antd';
import type { ButtonProps } from '../Button';
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
    /**
     * @deprecated Please use `actions` instead.
     */
    action?: AntDAlertProps['action'];
    /** Array of button props to render as footer actions. All buttons will be rendered as small size. */
    actions?: Omit<ButtonProps, 'size' | 'type'>[];
    /** Force actions to be rendered beneath the description. */
    forceVerticalActionsPlacement?: boolean;
    /** Content to display in a modal when "Show details" is clicked */
    showMoreContent?: React.ReactNode;
    /** Text for the "Show details" link. Defaults to "Show details" */
    showMoreText?: string;
    /** Text for "Show details" modal title. Defaults to "Details" */
    showMoreModalTitle?: string;
}
export declare const Alert: React.FC<AlertProps>;
//# sourceMappingURL=Alert.d.ts.map