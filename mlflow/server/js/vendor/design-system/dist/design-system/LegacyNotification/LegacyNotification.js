import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { notification as antDNotification } from 'antd';
import { forwardRef, useCallback, useMemo } from 'react';
import { DesignSystemAntDConfigProvider } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks';
import { useDesignSystemContext } from '../Hooks/useDesignSystemContext';
import { CloseIcon } from '../Icon';
import { SeverityIcon } from '../Icon/iconMap';
// Note: AntD only exposes context to notifications via the `useNotification` hook, and we need context to apply themes
// to AntD. As such you can currently only use notifications from within functional components.
/**
 * `useLegacyNotification` is deprecated in favor of the new `Notification` component.
 * @deprecated
 */
export function useLegacyNotification() {
    const [notificationInstance, contextHolder] = antDNotification.useNotification();
    const { getPrefixedClassName, theme } = useDesignSystemTheme();
    const { getPopupContainer: getContainer } = useDesignSystemContext();
    const clsPrefix = getPrefixedClassName('notification');
    const open = useCallback((args) => {
        const mergedArgs = {
            getContainer,
            ...defaultProps,
            ...args,
            style: {
                zIndex: theme.options.zIndexBase + 30,
                boxShadow: theme.general.shadowLow,
            },
        };
        const iconClassName = `${clsPrefix}-notice-icon-${mergedArgs.type}`;
        mergedArgs.icon = _jsx(SeverityIcon, { severity: mergedArgs.type, className: iconClassName });
        mergedArgs.closeIcon = (_jsx(CloseIcon, { "aria-hidden": "false", css: { cursor: 'pointer', fontSize: theme.general.iconSize }, "aria-label": mergedArgs.closeLabel || 'Close notification' }));
        notificationInstance.open(mergedArgs);
    }, [notificationInstance, getContainer, theme, clsPrefix]);
    const wrappedNotificationAPI = useMemo(() => {
        const error = (args) => open({ ...args, type: 'error' });
        const warning = (args) => open({ ...args, type: 'warning' });
        const info = (args) => open({ ...args, type: 'info' });
        const success = (args) => open({ ...args, type: 'success' });
        const close = (key) => antDNotification.close(key);
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
    return [wrappedNotificationAPI, _jsx(DesignSystemAntDConfigProvider, { children: contextHolder })];
}
const defaultProps = {
    type: 'info',
    duration: 3,
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
export const withNotifications = (Component) => forwardRef((props, ref) => {
    const [notificationAPI, notificationContextHolder] = useLegacyNotification();
    return (_jsx(Component, { ref: ref, notificationAPI: notificationAPI, notificationContextHolder: notificationContextHolder, ...props }));
});
//# sourceMappingURL=LegacyNotification.js.map