import { jsx as _jsx, Fragment as _Fragment, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { Switch as AntDSwitch } from 'antd';
import { useEffect, useMemo, useState } from 'react';
import { useDesignSystemEventComponentCallbacks, DesignSystemEventProviderAnalyticsEventTypes, DesignSystemEventProviderComponentTypes, } from '../DesignSystemEventProvider';
import { DesignSystemAntDConfigProvider, getAnimationCss } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks/useDesignSystemTheme';
import { Label } from '../Label/Label';
import { useDesignSystemSafexFlags, useNotifyOnFirstView } from '../utils';
import { importantify } from '../utils/css-utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
import { safex } from '../utils/safex';
import { useUniqueId } from '../utils/useUniqueId';
const getSwitchWithLabelStyles = ({ clsPrefix, theme, disabled, useNewShadows, }) => {
    // Default value
    const SWITCH_WIDTH = 28;
    const styles = {
        display: 'flex',
        alignItems: 'center',
        ...(disabled && {
            '&&, label': {
                color: theme.colors.actionDisabledText,
            },
        }),
        ...(useNewShadows && {
            [`&.${clsPrefix}-switch, &.${clsPrefix}-switch-checked`]: {
                [`.${clsPrefix}-switch-handle`]: {
                    top: -1,
                },
                [`.${clsPrefix}-switch-handle, .${clsPrefix}-switch-handle:before`]: {
                    width: 16,
                    height: 16,
                    borderRadius: theme.borders.borderRadiusFull,
                },
            },
        }),
        // Switch is Off
        [`&.${clsPrefix}-switch`]: {
            backgroundColor: theme.colors.backgroundPrimary,
            border: `1px solid ${theme.colors.actionDefaultBorderDefault}`,
            borderRadius: theme.borders.borderRadiusFull,
            [`.${clsPrefix}-switch-handle:before`]: {
                ...(useNewShadows
                    ? {
                        border: `1px solid ${theme.colors.actionDefaultBorderDefault}`,
                        boxShadow: theme.shadows.xs,
                        left: -1,
                    }
                    : {
                        boxShadow: `0px 0px 0px 1px ${theme.colors.actionDefaultBorderDefault}`,
                    }),
                transition: 'none',
                borderRadius: theme.borders.borderRadiusFull,
            },
            [`&:hover:not(.${clsPrefix}-switch-disabled)`]: {
                backgroundColor: theme.colors.actionDefaultBackgroundHover,
                border: `1px solid ${theme.colors.actionPrimaryBackgroundHover}`,
                [`.${clsPrefix}-switch-handle:before`]: {
                    ...(useNewShadows
                        ? {
                            boxShadow: theme.shadows.xs,
                            border: `1px solid ${theme.colors.actionPrimaryBackgroundHover}`,
                        }
                        : {
                            boxShadow: `0px 0px 0px 1px ${theme.colors.actionPrimaryBackgroundHover}`,
                        }),
                },
            },
            [`&:active:not(.${clsPrefix}-switch-disabled)`]: {
                backgroundColor: theme.colors.actionDefaultBackgroundPress,
                border: `1px solid ${theme.colors.actionPrimaryBackgroundPress}`,
                [`.${clsPrefix}-switch-handle:before`]: {
                    ...(useNewShadows
                        ? {
                            boxShadow: 'none',
                            border: `1px solid ${theme.colors.actionPrimaryBackgroundPress}`,
                        }
                        : {
                            boxShadow: `0px 0px 0px 1px ${theme.colors.actionPrimaryBackgroundHover}`,
                        }),
                },
            },
            [`&.${clsPrefix}-switch-disabled`]: {
                backgroundColor: theme.colors.actionDisabledBackground,
                border: `1px solid ${theme.colors.actionDisabledBorder}`,
                [`.${clsPrefix}-switch-handle:before`]: {
                    ...(useNewShadows
                        ? {
                            boxShadow: 'none',
                            border: `1px solid ${theme.colors.actionDisabledBorder}`,
                        }
                        : {
                            boxShadow: `0px 0px 0px 1px ${theme.colors.actionDisabledBorder}`,
                        }),
                },
            },
            [`&:focus-visible`]: {
                border: `1px solid ${theme.colors.actionPrimaryBackgroundDefault}`,
                boxShadow: 'none',
                outlineStyle: 'solid',
                outlineWidth: '1px',
                outlineColor: theme.colors.actionDefaultBorderFocus,
                [`.${clsPrefix}-switch-handle:before`]: {
                    ...(useNewShadows
                        ? {
                            boxShadow: theme.shadows.xs,
                            border: `1px solid ${theme.colors.actionPrimaryBackgroundDefault}`,
                        }
                        : {
                            boxShadow: `0px 0px 0px 1px ${theme.colors.actionPrimaryBackgroundDefault}`,
                        }),
                },
            },
            [`&:focus`]: {
                boxShadow: 'none',
            },
        },
        // Switch is On
        [`&.${clsPrefix}-switch-checked`]: {
            backgroundColor: theme.colors.actionPrimaryBackgroundDefault,
            border: `1px solid ${theme.colors.actionPrimaryBackgroundDefault}`,
            [`&:hover:not(.${clsPrefix}-switch-disabled)`]: {
                backgroundColor: theme.colors.actionPrimaryBackgroundHover,
                ...(useNewShadows
                    ? {
                        border: `1px solid ${theme.colors.actionPrimaryBackgroundHover}`,
                        [`.${clsPrefix}-switch-handle:before`]: {
                            border: `1px solid ${theme.colors.actionPrimaryBackgroundHover}`,
                        },
                    }
                    : {
                        border: `1px solid ${theme.colors.actionPrimaryBackgroundDefault}`,
                    }),
            },
            [`&:active:not(.${clsPrefix}-switch-disabled)`]: {
                backgroundColor: theme.colors.actionPrimaryBackgroundPress,
                ...(useNewShadows && {
                    border: `1px solid ${theme.colors.actionPrimaryBackgroundPress}`,
                    [`.${clsPrefix}-switch-handle:before`]: {
                        border: `1px solid ${theme.colors.actionPrimaryBackgroundPress}`,
                    },
                }),
            },
            [`.${clsPrefix}-switch-handle:before`]: {
                ...(useNewShadows
                    ? {
                        boxShadow: theme.shadows.xs,
                        border: `1px solid ${theme.colors.actionPrimaryBackgroundDefault}`,
                        right: -1,
                    }
                    : {
                        boxShadow: `0px 0px 0px 1px ${theme.colors.actionPrimaryBackgroundDefault}`,
                    }),
            },
            [`&.${clsPrefix}-switch-disabled`]: {
                backgroundColor: theme.colors.actionDisabledText,
                border: `1px solid ${theme.colors.actionDisabledText}`,
                [`.${clsPrefix}-switch-handle:before`]: {
                    ...(useNewShadows
                        ? {
                            boxShadow: 'none',
                            border: `1px solid ${theme.colors.actionDisabledText}`,
                        }
                        : {
                            boxShadow: `0px 0px 0px 1px ${theme.colors.actionDisabledText}`,
                        }),
                },
            },
            [`&:focus-visible`]: {
                outlineOffset: '1px',
            },
        },
        [`.${clsPrefix}-switch-handle:before`]: {
            backgroundColor: theme.colors.backgroundPrimary,
        },
        [`&& + .${clsPrefix}-hint, && + .${clsPrefix}-form-message`]: {
            paddingLeft: theme.spacing.sm + SWITCH_WIDTH,
        },
        [`&& + .${clsPrefix}-form-message`]: {
            marginTop: 0,
        },
        [`.${clsPrefix}-click-animating-node`]: {
            animation: 'none',
        },
        opacity: 1,
    };
    const importantStyles = importantify(styles);
    return css(importantStyles);
};
export const Switch = ({ dangerouslySetAntdProps, label, labelProps, activeLabel, inactiveLabel, disabledLabel, componentId, analyticsEvents, onChange, ...props }) => {
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.switch', false);
    const { theme, classNamePrefix } = useDesignSystemTheme();
    const duboisId = useUniqueId('dubois-switch');
    const { useNewShadows } = useDesignSystemSafexFlags();
    const uniqueId = props.id ?? duboisId;
    const [isChecked, setIsChecked] = useState(props.checked || props.defaultChecked);
    const memoizedAnalyticsEvents = useMemo(() => analyticsEvents ??
        (emitOnView
            ? [
                DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
                DesignSystemEventProviderAnalyticsEventTypes.OnView,
            ]
            : [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange]), [analyticsEvents, emitOnView]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.Switch,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        valueHasNoPii: true,
    });
    const { elementRef: viewRef } = useNotifyOnFirstView({
        onView: eventContext.onView,
        value: isChecked,
    });
    const handleToggle = (newState, event) => {
        eventContext.onValueChange(newState);
        if (onChange) {
            onChange(newState, event);
        }
        else {
            setIsChecked(newState);
        }
    };
    const onChangeHandler = (newState, event) => {
        eventContext.onValueChange(newState);
        onChange?.(newState, event);
    };
    useEffect(() => {
        setIsChecked(props.checked);
    }, [props.checked]);
    const hasNewLabels = activeLabel && inactiveLabel && disabledLabel;
    const stateMessage = isChecked ? activeLabel : inactiveLabel;
    // AntDSwitch's interface does not include `id` even though it passes it through and works as expected
    // We are using this to bypass that check
    const idPropObj = {
        id: uniqueId,
    };
    const switchComponent = (_jsx(AntDSwitch, { ...addDebugOutlineIfEnabled(), ...props, ...dangerouslySetAntdProps, onChange: handleToggle, ...idPropObj, css: {
            ...css(getAnimationCss(theme.options.enableAnimation)),
            ...getSwitchWithLabelStyles({
                clsPrefix: classNamePrefix,
                theme,
                disabled: props.disabled,
                useNewShadows,
            }),
        }, ref: viewRef }));
    const labelComponent = (_jsx(Label, { inline: true, ...labelProps, htmlFor: uniqueId, style: { ...(hasNewLabels && { marginRight: theme.spacing.sm }) }, children: label }));
    return label ? (_jsx(DesignSystemAntDConfigProvider, { children: _jsx("div", { ...addDebugOutlineIfEnabled(), css: getSwitchWithLabelStyles({
                clsPrefix: classNamePrefix,
                theme,
                disabled: props.disabled,
                useNewShadows,
            }), ...eventContext.dataComponentProps, children: hasNewLabels ? (_jsxs(_Fragment, { children: [labelComponent, _jsx("span", { style: { marginLeft: 'auto', marginRight: theme.spacing.sm, color: theme.colors.textPrimary }, children: `${stateMessage}${props.disabled ? ` (${disabledLabel})` : ''}` }), switchComponent] })) : (_jsxs(_Fragment, { children: [switchComponent, labelComponent] })) }) })) : (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(AntDSwitch, { onChange: onChangeHandler, ...addDebugOutlineIfEnabled(), ...props, ...dangerouslySetAntdProps, ...idPropObj, css: {
                ...css(getAnimationCss(theme.options.enableAnimation)),
                ...getSwitchWithLabelStyles({
                    clsPrefix: classNamePrefix,
                    theme,
                    disabled: props.disabled,
                    useNewShadows,
                }),
            }, ...eventContext.dataComponentProps, ref: viewRef }) }));
};
//# sourceMappingURL=Switch.js.map