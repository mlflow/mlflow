import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { keyframes, css } from '@emotion/react';
import * as Toast from '@radix-ui/react-toast';
import { forwardRef, useMemo } from 'react';
import { Button } from '../Button';
import { DesignSystemEventProviderComponentTypes, useDesignSystemEventComponentCallbacks, DesignSystemEventProviderAnalyticsEventTypes, DesignSystemEventProviderComponentSubTypeMap, } from '../DesignSystemEventProvider/DesignSystemEventProvider';
import { DU_BOIS_ENABLE_ANIMATION_CLASSNAME } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks';
import { CloseIcon } from '../Icon';
import { SeverityIcon } from '../Icon/iconMap';
import { getDarkModePortalStyles, useDesignSystemSafexFlags, useNotifyOnFirstView } from '../utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
const hideAnimation = keyframes({
    from: {
        opacity: 1,
    },
    to: {
        opacity: 0,
    },
});
const slideInAnimation = keyframes({
    from: {
        transform: 'translateX(calc(100% + 12px))',
    },
    to: {
        transform: 'translateX(0)',
    },
});
const swipeOutAnimation = keyframes({
    from: {
        transform: 'translateX(var(--radix-toast-swipe-end-x))',
    },
    to: {
        transform: 'translateX(calc(100% + 12px))',
    },
});
const getToastRootStyle = (theme, classNamePrefix, useNewShadows, useNewBorderColors) => {
    return css({
        '&&': {
            position: 'relative',
            display: 'grid',
            background: theme.colors.backgroundPrimary,
            padding: 12,
            columnGap: 4,
            boxShadow: useNewShadows ? theme.shadows.lg : theme.general.shadowLow,
            borderRadius: theme.borders.borderRadiusSm,
            lineHeight: '20px',
            ...(useNewBorderColors && {
                borderColor: `1px solid ${theme.colors.border}`,
            }),
            gridTemplateRows: '[header] auto [content] auto',
            gridTemplateColumns: '[icon] auto [content] 1fr [close] auto',
            ...getDarkModePortalStyles(theme, useNewShadows, useNewBorderColors),
        },
        [`.${classNamePrefix}-notification-severity-icon`]: {
            gridRow: 'header / content',
            gridColumn: 'icon / icon',
            display: 'inline-flex',
            alignItems: 'center',
        },
        [`.${classNamePrefix}-btn`]: {
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
        },
        [`.${classNamePrefix}-notification-info-icon`]: {
            color: theme.colors.textSecondary,
        },
        [`.${classNamePrefix}-notification-success-icon`]: {
            color: theme.colors.textValidationSuccess,
        },
        [`.${classNamePrefix}-notification-warning-icon`]: {
            color: theme.colors.textValidationWarning,
        },
        [`.${classNamePrefix}-notification-error-icon`]: {
            color: theme.colors.textValidationDanger,
        },
        '&&[data-state="open"]': {
            animation: `${slideInAnimation} 300ms cubic-bezier(0.16, 1, 0.3, 1)`,
        },
        '&[data-state="closed"]': {
            animation: `${hideAnimation} 100ms ease-in`,
        },
        '&[data-swipe="move"]': {
            transform: 'translateX(var(--radix-toast-swipe-move-x))',
        },
        '&[data-swipe="cancel"]': {
            transform: 'translateX(0)',
            transition: 'transform 200ms ease-out',
        },
        '&[data-swipe="end"]': {
            animation: `${swipeOutAnimation} 100ms ease-out`,
        },
    });
};
export const Root = forwardRef(function ({ children, severity = 'info', componentId, analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnView], ...props }, ref) {
    const { theme, classNamePrefix } = useDesignSystemTheme();
    const { useNewShadows, useNewBorderColors } = useDesignSystemSafexFlags();
    const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.Notification,
        componentId,
        componentSubType: DesignSystemEventProviderComponentSubTypeMap[severity],
        analyticsEvents: memoizedAnalyticsEvents,
        shouldStartInteraction: false,
    });
    // A new ref was created rather than creating additional complexity of merging the refs, something to consider for the future to optimize
    const { elementRef } = useNotifyOnFirstView({ onView: eventContext.onView });
    return (_jsxs(Toast.Root, { ref: ref, css: getToastRootStyle(theme, classNamePrefix, useNewShadows, useNewBorderColors), ...props, ...addDebugOutlineIfEnabled(), children: [_jsx(SeverityIcon, { className: `${classNamePrefix}-notification-severity-icon ${classNamePrefix}-notification-${severity}-icon`, severity: severity, ref: elementRef }), children] }));
});
// TODO: Support light and dark mode
const getViewportStyle = (theme) => {
    return {
        position: 'fixed',
        top: 0,
        right: 0,
        display: 'flex',
        flexDirection: 'column',
        padding: 12,
        gap: 12,
        width: 440,
        listStyle: 'none',
        zIndex: theme.options.zIndexBase + 100,
        outline: 'none',
        maxWidth: `calc(100% - ${theme.spacing.lg}px)`,
    };
};
const getTitleStyles = (theme) => {
    return css({
        fontWeight: theme.typography.typographyBoldFontWeight,
        color: theme.colors.textPrimary,
        gridRow: 'header / header',
        gridColumn: 'content / content',
        userSelect: 'text',
    });
};
export const Title = forwardRef(function ({ children, ...props }, ref) {
    const { theme } = useDesignSystemTheme();
    return (_jsx(Toast.Title, { ref: ref, css: getTitleStyles(theme), ...props, children: children }));
});
const getDescriptionStyles = (theme) => {
    return css({
        marginTop: 4,
        color: theme.colors.textPrimary,
        gridRow: 'content / content',
        gridColumn: 'content / content',
        userSelect: 'text',
    });
};
export const Description = forwardRef(function ({ children, ...props }, ref) {
    const { theme } = useDesignSystemTheme();
    return (_jsx(Toast.Description, { ref: ref, css: getDescriptionStyles(theme), ...props, children: children }));
});
const getCloseStyles = (theme) => {
    return css({
        color: theme.colors.textSecondary,
        position: 'absolute',
        // Offset close button position to align with the title, title uses 20px line height, button has 32px
        right: 6,
        top: 6,
    });
};
export const Close = forwardRef(function (props, ref) {
    const { theme } = useDesignSystemTheme();
    const { closeLabel, componentId, analyticsEvents, ...restProps } = props;
    return (
    // Wrapper to keep close column width for content sizing, close button positioned absolute for alignment without affecting the grid's first row height (title)
    _jsx("div", { style: { gridColumn: 'close / close', gridRow: 'header / content', width: 20 }, children: _jsx(Toast.Close, { ref: ref, css: getCloseStyles(theme), ...restProps, asChild: true, children: _jsx(Button, { componentId: componentId ? componentId : 'codegen_design-system_src_design-system_notification_notification.tsx_224', analyticsEvents: analyticsEvents, icon: _jsx(CloseIcon, {}), "aria-label": closeLabel ?? restProps['aria-label'] ?? 'Close notification' }) }) }));
});
export const Provider = ({ children, ...props }) => {
    return _jsx(Toast.Provider, { ...props, children: children });
};
export const Viewport = (props) => {
    const { theme } = useDesignSystemTheme();
    return _jsx(Toast.Viewport, { className: DU_BOIS_ENABLE_ANIMATION_CLASSNAME, style: getViewportStyle(theme), ...props });
};
//# sourceMappingURL=Notification.js.map