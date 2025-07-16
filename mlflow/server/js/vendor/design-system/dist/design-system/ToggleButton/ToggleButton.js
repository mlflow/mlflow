import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { useMergeRefs } from '@floating-ui/react';
import * as Toggle from '@radix-ui/react-toggle';
import React, { forwardRef, useCallback, useEffect, useMemo } from 'react';
import { DesignSystemEventProviderAnalyticsEventTypes, DesignSystemEventProviderComponentTypes, useDesignSystemEventComponentCallbacks, } from '../DesignSystemEventProvider';
import { useDesignSystemTheme } from '../Hooks';
import { CheckIcon, Icon } from '../Icon';
import { useDesignSystemSafexFlags } from '../utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
import { safex } from '../utils/safex';
import { useNotifyOnFirstView } from '../utils/useNotifyOnFirstView';
const SMALL_BUTTON_HEIGHT = 24;
const getStyles = (theme, size, onlyIcon, useNewShadows) => {
    return css({
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        whiteSpace: 'nowrap',
        ...(useNewShadows && {
            boxShadow: theme.shadows.xs,
        }),
        border: `1px solid ${theme.colors.actionDefaultBorderDefault}`,
        borderRadius: theme.borders.borderRadiusSm,
        backgroundColor: 'transparent',
        color: theme.colors.actionDefaultTextDefault,
        height: theme.general.heightSm,
        padding: '0 12px',
        fontSize: theme.typography.fontSizeBase,
        lineHeight: `${theme.typography.lineHeightBase}px`,
        '&[data-state="off"] .togglebutton-icon-wrapper': {
            color: theme.colors.textSecondary,
        },
        '&[data-state="off"]:hover .togglebutton-icon-wrapper': {
            color: theme.colors.actionDefaultTextHover,
        },
        '&[data-state="on"]': {
            backgroundColor: theme.colors.actionDefaultBackgroundPress,
            color: theme.colors.actionDefaultTextPress,
            borderColor: theme.colors.actionDefaultBorderPress,
        },
        '&:hover': {
            cursor: 'pointer',
            color: theme.colors.actionDefaultTextHover,
            backgroundColor: theme.colors.actionDefaultBackgroundHover,
            borderColor: theme.colors.actionDefaultBorderHover,
            '& > svg': {
                stroke: theme.colors.actionDefaultBorderHover,
            },
        },
        '&:disabled': {
            cursor: 'default',
            borderColor: theme.colors.actionDisabledBorder,
            color: theme.colors.actionDisabledText,
            backgroundColor: 'transparent',
            ...(useNewShadows && {
                boxShadow: 'none',
            }),
            '& > svg': {
                stroke: theme.colors.border,
            },
        },
        ...(!onlyIcon && {
            '&&': {
                padding: '4px 12px',
                ...(size === 'small' && {
                    padding: '0 8px',
                }),
            },
        }),
        ...(onlyIcon && {
            width: theme.general.heightSm,
            border: 'none',
        }),
        ...(size === 'small' && {
            height: SMALL_BUTTON_HEIGHT,
            lineHeight: theme.typography.lineHeightBase,
            ...(onlyIcon && {
                width: SMALL_BUTTON_HEIGHT,
                paddingTop: 0,
                paddingBottom: 0,
                verticalAlign: 'middle',
            }),
        }),
    });
};
const RectangleSvg = (props) => (_jsx("svg", { width: "16", height: "16", viewBox: "0 0 16 16", fill: "none", xmlns: "http://www.w3.org/2000/svg", ...props, children: _jsx("rect", { x: "0.5", y: "0.5", width: "15", height: "15", rx: "3.5" }) }));
const RectangleIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: RectangleSvg });
});
export const ToggleButton = forwardRef(({ children, pressed, defaultPressed, icon, size = 'middle', componentId, analyticsEvents, ...props }, ref) => {
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.toggleButton', false);
    const { theme } = useDesignSystemTheme();
    const { useNewShadows } = useDesignSystemSafexFlags();
    const [isPressed, setIsPressed] = React.useState(defaultPressed);
    const memoizedAnalyticsEvents = useMemo(() => analyticsEvents ??
        (emitOnView
            ? [
                DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
                DesignSystemEventProviderAnalyticsEventTypes.OnView,
            ]
            : [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange]), [analyticsEvents, emitOnView]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.ToggleButton,
        componentId: componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        valueHasNoPii: true,
    });
    const { elementRef: toggleButtonRef } = useNotifyOnFirstView({
        onView: eventContext.onView,
        value: pressed ?? defaultPressed,
    });
    const mergedRef = useMergeRefs([ref, toggleButtonRef]);
    const handleOnPressedChange = useCallback((pressed) => {
        eventContext.onValueChange(pressed);
        props.onPressedChange?.(pressed);
        setIsPressed(pressed);
    }, [eventContext, props]);
    useEffect(() => {
        setIsPressed(pressed);
    }, [pressed]);
    const iconOnly = !children && Boolean(icon);
    const iconStyle = iconOnly ? {} : { marginRight: theme.spacing.xs };
    const checkboxIcon = isPressed ? (_jsx(CheckIcon, {})) : (_jsx(RectangleIcon, { css: {
            stroke: theme.colors.border,
        } }));
    return (_jsxs(Toggle.Root, { ...addDebugOutlineIfEnabled(), css: getStyles(theme, size, iconOnly, useNewShadows), ...props, pressed: isPressed, onPressedChange: handleOnPressedChange, ref: mergedRef, ...eventContext.dataComponentProps, children: [_jsx("span", { className: "togglebutton-icon-wrapper", style: { display: 'flex', ...iconStyle }, children: icon ? icon : checkboxIcon }), children] }));
});
//# sourceMappingURL=ToggleButton.js.map