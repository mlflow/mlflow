import type { AlertProps as AntDAlertProps } from 'antd';
import type { ButtonProps } from '../Button';
import { DesignSystemEventProviderAnalyticsEventTypes } from '../DesignSystemEventProvider';
import type { AnalyticsEventProps, DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
export type AlertType = NonNullable<AntDAlertProps['type']>;
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
    /**
     * When true (and `forceVerticalActionsPlacement` is false), multiple `actions` render in the top
     * horizontal action slot instead of stacking under the description.
     */
    forceHorizontalActionsPlacement?: boolean;
    /** When true, the description body (and vertical actions, if any) can be collapsed via a chevron control. */
    collapsible?: boolean;
    /** Initial expanded state when `collapsible` is true. Defaults to true. */
    defaultExpanded?: boolean;
    /** Accessible label for the expand control when collapsed. Defaults to "Expand alert details". */
    expandToggleAriaLabel?: string;
    /** Accessible label for the collapse control when expanded. Defaults to "Collapse alert details". */
    collapseToggleAriaLabel?: string;
    /** Content to display in a modal when "Show details" is clicked */
    showMoreContent?: React.ReactNode;
    /** Text for the "Show details" link. Defaults to "Show details" */
    showMoreText?: string;
    /** Text for "Show details" modal title. Defaults to "Details" */
    showMoreModalTitle?: string;
    size?: 'small' | 'large';
}
export declare const Alert: React.FC<React.PropsWithChildren<AlertProps>>;
//# sourceMappingURL=Alert.d.ts.map