import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { useMergeRefs } from '@floating-ui/react';
import { RadioGroup, RadioGroupItem } from '@radix-ui/react-radio-group';
import React, { useCallback, useMemo } from 'react';
import { DesignSystemEventProviderAnalyticsEventTypes, DesignSystemEventProviderComponentTypes, useDesignSystemEventComponentCallbacks, useDesignSystemSafexFlags, useNotifyOnFirstView, useDesignSystemTheme, } from '../../design-system';
const RadioGroupContext = React.createContext('medium');
export const Root = React.forwardRef(({ size, componentId, analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange], valueHasNoPii, onValueChange, ...props }, forwardedRef) => {
    const { theme } = useDesignSystemTheme();
    const contextValue = React.useMemo(() => size ?? 'medium', [size]);
    const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.PillControl,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        valueHasNoPii,
    });
    const { elementRef: pillControlRef } = useNotifyOnFirstView({
        onView: eventContext.onView,
        value: props.value ?? props.defaultValue,
    });
    const onValueChangeWrapper = useCallback((value) => {
        eventContext.onValueChange?.(value);
        onValueChange?.(value);
    }, [eventContext, onValueChange]);
    const mergedRef = useMergeRefs([forwardedRef, pillControlRef]);
    return (_jsx(RadioGroupContext.Provider, { value: contextValue, children: _jsx(RadioGroup, { css: {
                display: 'flex',
                flexWrap: 'wrap',
                gap: theme.spacing.sm,
            }, onValueChange: onValueChangeWrapper, ...props, ref: mergedRef, ...eventContext.dataComponentProps }) }));
});
export const Item = React.forwardRef(({ children, icon, ...props }, forwardedRef) => {
    const size = React.useContext(RadioGroupContext);
    const { theme } = useDesignSystemTheme();
    const iconClass = 'pill-control-icon';
    const css = useRadioGroupItemStyles(size, iconClass);
    return (_jsxs(RadioGroupItem, { css: css, ...props, children: [icon && (_jsx("span", { className: iconClass, css: {
                    marginRight: size === 'large' ? theme.spacing.sm : theme.spacing.xs,
                    [`& > .anticon`]: { verticalAlign: `-3px` },
                }, children: icon })), children] }));
});
const useRadioGroupItemStyles = (size, iconClass) => {
    const { theme } = useDesignSystemTheme();
    const { useNewShadows, useNewBorderColors } = useDesignSystemSafexFlags();
    return {
        textOverflow: 'ellipsis',
        ...(useNewShadows
            ? {
                boxShadow: theme.shadows.xs,
            }
            : {}),
        whiteSpace: 'nowrap',
        overflow: 'hidden',
        appearance: 'none',
        textDecoration: 'none',
        background: 'none',
        border: '1px solid',
        cursor: 'pointer',
        backgroundColor: theme.colors.actionDefaultBackgroundDefault,
        borderColor: useNewBorderColors ? theme.colors.actionDefaultBorderDefault : theme.colors.border,
        color: theme.colors.textPrimary,
        lineHeight: theme.typography.lineHeightBase,
        height: 32,
        paddingInline: '12px',
        fontWeight: theme.typography.typographyRegularFontWeight,
        fontSize: theme.typography.fontSizeBase,
        borderRadius: theme.borders.borderRadiusFull,
        transition: 'background-color 0.2s ease-in-out, border-color 0.2s ease-in-out',
        [`& > .${iconClass}`]: {
            color: theme.colors.textSecondary,
            ...(size === 'large'
                ? {
                    backgroundColor: theme.colors.tagDefault,
                    padding: theme.spacing.sm,
                    borderRadius: theme.borders.borderRadiusFull,
                }
                : {}),
        },
        '&[data-state="checked"]': {
            backgroundColor: theme.colors.actionDefaultBackgroundPress,
            borderColor: 'transparent',
            color: theme.colors.textPrimary,
            // outline
            outlineStyle: 'solid',
            outlineWidth: '2px',
            outlineOffset: '0px',
            outlineColor: theme.colors.actionDefaultBorderFocus,
            '&:hover': {
                backgroundColor: theme.colors.actionDefaultBackgroundPress,
                borderColor: theme.colors.actionLinkPress,
                color: 'inherit',
            },
            [`& > .${iconClass}, &:hover > .${iconClass}`]: {
                color: theme.colors.actionDefaultTextPress,
                ...(size === 'large'
                    ? {
                        backgroundColor: theme.colors.actionIconBackgroundPress,
                    }
                    : {}),
            },
        },
        '&:focus-visible': {
            outlineStyle: 'solid',
            outlineWidth: '2px',
            outlineOffset: '0px',
            outlineColor: theme.colors.actionDefaultBorderFocus,
        },
        '&:hover': {
            backgroundColor: theme.colors.actionDefaultBackgroundHover,
            borderColor: theme.colors.actionLinkHover,
            color: theme.colors.actionDefaultTextHover,
            [`& > .${iconClass}`]: {
                color: 'inherit',
                ...(size === 'large'
                    ? {
                        backgroundColor: theme.colors.actionIconBackgroundHover,
                    }
                    : {}),
            },
        },
        '&:active': {
            backgroundColor: theme.colors.actionDefaultBackgroundPress,
            borderColor: theme.colors.actionLinkPress,
            color: theme.colors.actionDefaultTextPress,
            [`& > .${iconClass}`]: {
                color: 'inherit',
                ...(size === 'large'
                    ? {
                        backgroundColor: theme.colors.actionIconBackgroundPress,
                    }
                    : {}),
            },
        },
        '&:disabled': {
            backgroundColor: theme.colors.actionDisabledBackground,
            borderColor: theme.colors.actionDisabledBorder,
            color: theme.colors.actionDisabledText,
            cursor: 'not-allowed',
            [`& > .${iconClass}`]: {
                color: 'inherit',
            },
        },
        ...(size === 'small'
            ? {
                height: 24,
                lineHeight: theme.typography.lineHeightSm,
                paddingInline: theme.spacing.sm,
            }
            : {}),
        ...(size === 'large'
            ? {
                height: 44,
                lineHeight: theme.typography.lineHeightXl,
                paddingInline: theme.spacing.md,
                paddingInlineStart: '6px',
                borderRadius: theme.borders.borderRadiusFull,
            }
            : {}),
    };
};
//# sourceMappingURL=PillControl.js.map