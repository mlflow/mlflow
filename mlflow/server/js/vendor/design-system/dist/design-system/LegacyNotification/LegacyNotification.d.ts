import type { ArgsProps as AntDNotificationArgs, IconType as AntDIconType, NotificationInstance as AntDNotificationInstance } from 'antd/lib/notification';
import type { DangerouslySetAntdProps } from '../types';
export type NotificationType = AntDIconType;
export interface NotificationInstance extends AntDNotificationInstance {
    close: (key: string) => void;
}
/**
 * `LegacyNotification` is deprecated in favor of the new `Notification` component.
 * @deprecated
 */
export interface LegacyNotificationProps extends Omit<AntDNotificationArgs, 'bottom' | 'btn' | 'className' | 'closeIcon' | 'icon' | 'position' | 'style' | 'top'>, DangerouslySetAntdProps<AntDNotificationArgs> {
    type: NotificationType;
    closeLabel?: string;
}
/**
 * `useLegacyNotification` is deprecated in favor of the new `Notification` component.
 * @deprecated
 */
export declare function useLegacyNotification(): [NotificationInstance, React.ReactElement];
/**
 * A type wrapping given component interface with props returned by withNotifications() HOC
 *
 * @deprecated Please migrate components to functional components and use useNotification() hook instead.
 */
export type WithNotificationsHOCProps<T> = T & {
    notificationAPI: NotificationInstance;
    notificationContextHolder: React.ReactElement;
};
/**
 * A higher-order component factory function, enables using notifications in
 * class components in a similar way to useNotification() hook. Wrapped component will have
 * additional "notificationAPI" and "notificationContextHolder" props injected containing
 * the notification API object and context holder react node respectively.
 *
 * The wrapped component can implement WithNotificationsHOCProps<OwnProps> type which
 * enriches the component's interface with the mentioned props.
 *
 * @deprecated Please migrate components to functional components and use useNotification() hook instead.
 */
export declare const withNotifications: <P, T>(Component: React.ComponentType<WithNotificationsHOCProps<P>>) => import("react").ForwardRefExoticComponent<import("react").PropsWithoutRef<P> & import("react").RefAttributes<T>>;
//# sourceMappingURL=LegacyNotification.d.ts.map