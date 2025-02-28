import { notification as antDNotification } from 'antd';
import type {
  ArgsProps as AntDNotificationArgs,
  IconType as AntDIconType,
  NotificationInstance as AntDNotificationInstance,
} from 'antd/lib/notification';
import { forwardRef, useCallback, useMemo } from 'react';

import { DesignSystemAntDConfigProvider } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks';
import { useDesignSystemContext } from '../Hooks/useDesignSystemContext';
import { CloseIcon } from '../Icon';
import { SeverityIcon } from '../Icon/iconMap';
import type { DangerouslySetAntdProps } from '../types';

export type NotificationType = AntDIconType;
export interface NotificationInstance extends AntDNotificationInstance {
  close: (key: string) => void;
}

/**
 * `LegacyNotification` is deprecated in favor of the new `Notification` component.
 * @deprecated
 */
export interface LegacyNotificationProps
  extends Omit<
      AntDNotificationArgs,
      'bottom' | 'btn' | 'className' | 'closeIcon' | 'icon' | 'position' | 'style' | 'top'
    >,
    DangerouslySetAntdProps<AntDNotificationArgs> {
  type: NotificationType;
  closeLabel?: string;
}

// Note: AntD only exposes context to notifications via the `useNotification` hook, and we need context to apply themes
// to AntD. As such you can currently only use notifications from within functional components.
/**
 * `useLegacyNotification` is deprecated in favor of the new `Notification` component.
 * @deprecated
 */
export function useLegacyNotification(): [NotificationInstance, React.ReactElement] {
  const [notificationInstance, contextHolder] = antDNotification.useNotification();
  const { getPrefixedClassName, theme } = useDesignSystemTheme();
  const { getPopupContainer: getContainer } = useDesignSystemContext();

  const clsPrefix = getPrefixedClassName('notification');

  const open = useCallback(
    (args: LegacyNotificationProps) => {
      const mergedArgs: AntDNotificationArgs & LegacyNotificationProps = {
        getContainer,
        ...defaultProps,
        ...args,
        style: {
          zIndex: theme.options.zIndexBase + 30,
          boxShadow: theme.general.shadowLow,
        },
      };

      const iconClassName = `${clsPrefix}-notice-icon-${mergedArgs.type}`;
      mergedArgs.icon = <SeverityIcon severity={mergedArgs.type} className={iconClassName} />;

      mergedArgs.closeIcon = (
        <CloseIcon
          aria-hidden="false"
          css={{ cursor: 'pointer', fontSize: theme.general.iconSize }}
          aria-label={mergedArgs.closeLabel || 'Close notification'}
        />
      );

      notificationInstance.open(mergedArgs);
    },
    [notificationInstance, getContainer, theme, clsPrefix],
  );

  const wrappedNotificationAPI: NotificationInstance = useMemo(() => {
    const error = (args: LegacyNotificationProps) => open({ ...args, type: 'error' });
    const warning = (args: LegacyNotificationProps) => open({ ...args, type: 'warning' });
    const info = (args: LegacyNotificationProps) => open({ ...args, type: 'info' });
    const success = (args: LegacyNotificationProps) => open({ ...args, type: 'success' });
    const close = (key: string) => antDNotification.close(key);

    return {
      open,
      close,
      error,
      warning,
      info,
      success,
    };
  }, [open]);

  // eslint-disable-next-line react/jsx-key -- TODO(FEINF-1756)
  return [wrappedNotificationAPI, <DesignSystemAntDConfigProvider>{contextHolder}</DesignSystemAntDConfigProvider>];
}

const defaultProps: Partial<LegacyNotificationProps> = {
  type: 'info',
  duration: 3,
};

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
export const withNotifications = <P, T>(Component: React.ComponentType<WithNotificationsHOCProps<P>>) =>
  forwardRef<T, P>((props: P, ref) => {
    const [notificationAPI, notificationContextHolder] = useLegacyNotification();
    return (
      <Component
        ref={ref}
        notificationAPI={notificationAPI}
        notificationContextHolder={notificationContextHolder}
        {...props}
      />
    );
  });
