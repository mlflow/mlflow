import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { Checkbox as AntDCheckbox } from 'antd';
import classnames from 'classnames';
import { forwardRef, useMemo } from 'react';
import { DesignSystemEventProviderAnalyticsEventTypes, DesignSystemEventProviderComponentTypes, useDesignSystemEventComponentCallbacks, } from '../DesignSystemEventProvider';
import { DesignSystemAntDConfigProvider, getAnimationCss, RestoreAntDDefaultClsPrefix } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks/useDesignSystemTheme';
import { useDesignSystemSafexFlags, useNotifyOnFirstView } from '../utils';
import { importantify } from '../utils/css-utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
import { safex } from '../utils/safex';
function getCheckboxEmotionStyles(clsPrefix, theme, isHorizontal = false, useNewShadows, useNewFormUISpacing, useNewBorderRadii) {
    const classInput = `.${clsPrefix}-input`;
    const classInner = `.${clsPrefix}-inner`;
    const classIndeterminate = `.${clsPrefix}-indeterminate`;
    const classChecked = `.${clsPrefix}-checked`;
    const classDisabled = `.${clsPrefix}-disabled`;
    const classDisabledWrapper = `.${clsPrefix}-wrapper-disabled`;
    const classContainer = `.${clsPrefix}-group`;
    const classWrapper = `.${clsPrefix}-wrapper`;
    const defaultSelector = `${classInput} + ${classInner}`;
    const hoverSelector = `${classInput}:hover + ${classInner}`;
    const pressSelector = `${classInput}:active + ${classInner}`;
    const cleanClsPrefix = `.${clsPrefix.replace('-checkbox', '')}`;
    const styles = {
        [`.${clsPrefix}`]: {
            top: 'unset',
            lineHeight: theme.typography.lineHeightBase,
            alignSelf: 'flex-start',
            display: 'flex',
            alignItems: 'center',
            height: theme.typography.lineHeightBase,
        },
        [`&${classWrapper}, ${classWrapper}`]: {
            alignItems: 'center',
            lineHeight: theme.typography.lineHeightBase,
        },
        // Top level styles are for the unchecked state
        [classInner]: {
            borderColor: theme.colors.actionDefaultBorderDefault,
            ...(useNewBorderRadii && {
                borderRadius: theme.borders.borderRadiusSm,
            }),
        },
        // Style wrapper span added by Antd
        [`&> span:not(.${clsPrefix})`]: {
            display: 'inline-flex',
            alignItems: 'center',
        },
        // Layout styling
        [`&${classContainer}`]: {
            display: 'flex',
            flexDirection: 'column',
            rowGap: theme.spacing.sm,
            columnGap: 0,
            ...(useNewFormUISpacing && {
                [`& + ${cleanClsPrefix}-form-message`]: {
                    marginTop: theme.spacing.sm,
                },
            }),
        },
        ...(useNewFormUISpacing && {
            [`${cleanClsPrefix}-hint + &${classContainer}`]: {
                marginTop: theme.spacing.sm,
            },
        }),
        ...(isHorizontal && {
            [`&${classContainer}`]: {
                display: 'flex',
                flexDirection: 'row',
                columnGap: theme.spacing.sm,
                rowGap: 0,
                [`& > ${classContainer}-item`]: {
                    marginRight: 0,
                },
            },
        }),
        // Keyboard focus
        [`${classInput}:focus-visible + ${classInner}`]: {
            outlineWidth: '2px',
            outlineColor: theme.colors.actionDefaultBorderFocus,
            outlineOffset: '4px',
            outlineStyle: 'solid',
        },
        // Hover
        [hoverSelector]: {
            backgroundColor: theme.colors.actionDefaultBackgroundHover,
            borderColor: theme.colors.actionPrimaryBackgroundHover,
        },
        // Mouse pressed
        [pressSelector]: {
            backgroundColor: theme.colors.actionDefaultBackgroundPress,
            borderColor: theme.colors.actionPrimaryBackgroundPress,
        },
        // Checked state
        [classChecked]: {
            ...(useNewShadows && {
                [classInner]: {
                    boxShadow: theme.shadows.xs,
                },
            }),
            '&::after': {
                border: 'none',
            },
            [defaultSelector]: {
                backgroundColor: theme.colors.actionPrimaryBackgroundDefault,
                borderColor: 'transparent',
            },
            // Checked hover
            [hoverSelector]: {
                backgroundColor: theme.colors.actionPrimaryBackgroundHover,
                borderColor: 'transparent',
            },
            // Checked and mouse pressed
            [pressSelector]: {
                backgroundColor: theme.colors.actionPrimaryBackgroundPress,
                borderColor: 'transparent',
            },
        },
        // Indeterminate
        [classIndeterminate]: {
            [classInner]: {
                ...(useNewShadows && {
                    boxShadow: theme.shadows.xs,
                }),
                backgroundColor: theme.colors.actionPrimaryBackgroundDefault,
                borderColor: theme.colors.actionPrimaryBackgroundDefault,
                // The after pseudo-element is used for the check image itself
                '&:after': {
                    backgroundColor: theme.colors.white,
                    height: '3px',
                    width: '8px',
                    borderRadius: '4px',
                },
            },
            // Indeterminate hover
            [hoverSelector]: {
                backgroundColor: theme.colors.actionPrimaryBackgroundHover,
                borderColor: 'transparent',
            },
            // Indeterminate and mouse pressed
            [pressSelector]: {
                backgroundColor: theme.colors.actionPrimaryBackgroundPress,
            },
        },
        // Disabled state
        [`&${classDisabledWrapper}`]: {
            [classDisabled]: {
                // Disabled Checked
                [`&${classChecked}`]: {
                    [classInner]: {
                        backgroundColor: theme.colors.actionDisabledBackground,
                        borderColor: theme.colors.actionDisabledBorder,
                        '&:after': {
                            borderColor: theme.colors.actionDisabledText,
                        },
                    },
                    // Disabled checked hover
                    [hoverSelector]: {
                        backgroundColor: theme.colors.actionDisabledBackground,
                        borderColor: theme.colors.actionDisabledBorder,
                    },
                },
                // Disabled indeterminate
                [`&${classIndeterminate}`]: {
                    [classInner]: {
                        backgroundColor: theme.colors.actionDisabledBackground,
                        borderColor: theme.colors.actionDisabledBorder,
                        '&:after': {
                            borderColor: theme.colors.actionDisabledText,
                            backgroundColor: theme.colors.actionDisabledText,
                        },
                    },
                    // Disabled indeterminate hover
                    [hoverSelector]: {
                        backgroundColor: theme.colors.actionDisabledBackground,
                        borderColor: theme.colors.actionDisabledBorder,
                    },
                },
                // Disabled unchecked
                [classInner]: {
                    backgroundColor: theme.colors.actionDisabledBackground,
                    borderColor: theme.colors.actionDisabledBorder,
                    // The after pseudo-element is used for the check image itself
                    '&:after': {
                        borderColor: 'transparent',
                    },
                },
                // Disabled hover
                [hoverSelector]: {
                    backgroundColor: theme.colors.actionDisabledBackground,
                    borderColor: theme.colors.actionDisabledBorder,
                },
                '& + span': {
                    color: theme.colors.actionDisabledText,
                },
            },
        },
        // Animation
        ...getAnimationCss(theme.options.enableAnimation),
    };
    return styles;
}
export const getWrapperStyle = ({ clsPrefix, theme, wrapperStyle = {}, useNewFormUISpacing, }) => {
    const extraSelector = useNewFormUISpacing ? `, && + .${clsPrefix}-hint + .${clsPrefix}-form-message` : '';
    const styles = {
        height: theme.typography.lineHeightBase,
        lineHeight: theme.typography.lineHeightBase,
        [`&& + .${clsPrefix}-hint, && + .${clsPrefix}-form-message${extraSelector}`]: {
            paddingLeft: theme.spacing.lg,
            marginTop: 0,
        },
        ...wrapperStyle,
    };
    return css(styles);
};
const DuboisCheckbox = forwardRef(function Checkbox({ isChecked, onChange, children, isDisabled = false, style, wrapperStyle, dangerouslySetAntdProps, className, componentId, analyticsEvents, ...restProps }, ref) {
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.checkbox', false);
    const { theme, classNamePrefix, getPrefixedClassName } = useDesignSystemTheme();
    const { useNewShadows, useNewFormUISpacing, useNewBorderRadii } = useDesignSystemSafexFlags();
    const clsPrefix = getPrefixedClassName('checkbox');
    const memoizedAnalyticsEvents = useMemo(() => analyticsEvents ??
        (emitOnView
            ? [
                DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
                DesignSystemEventProviderAnalyticsEventTypes.OnView,
            ]
            : [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange]), [analyticsEvents, emitOnView]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.Checkbox,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        valueHasNoPii: true,
    });
    const { elementRef: checkboxRef } = useNotifyOnFirstView({
        onView: eventContext.onView,
        value: isChecked ?? restProps.defaultChecked,
    });
    const onChangeHandler = (event) => {
        eventContext.onValueChange(event.target.checked);
        onChange?.(event.target.checked, event);
    };
    return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx("div", { ...addDebugOutlineIfEnabled(), className: classnames(className, `${clsPrefix}-container`), css: getWrapperStyle({
                clsPrefix: classNamePrefix,
                theme,
                wrapperStyle,
                useNewFormUISpacing,
            }), ref: checkboxRef, children: _jsx(AntDCheckbox, { checked: isChecked === null ? undefined : isChecked, ref: ref, onChange: onChangeHandler, disabled: isDisabled, indeterminate: isChecked === null, 
                // Individual checkboxes don't depend on isHorizontal flag, orientation and spacing is handled by end users
                css: css(importantify(getCheckboxEmotionStyles(clsPrefix, theme, false, useNewShadows, useNewFormUISpacing, useNewBorderRadii))), style: style, "aria-checked": isChecked === null ? 'mixed' : isChecked, ...restProps, ...dangerouslySetAntdProps, ...eventContext.dataComponentProps, children: _jsx(RestoreAntDDefaultClsPrefix, { children: children }) }) }) }));
});
const CheckboxGroup = forwardRef(function CheckboxGroup({ children, layout = 'vertical', ...props }, ref) {
    const { theme, getPrefixedClassName } = useDesignSystemTheme();
    const clsPrefix = getPrefixedClassName('checkbox');
    const { useNewShadows, useNewFormUISpacing, useNewBorderRadii } = useDesignSystemSafexFlags();
    return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(AntDCheckbox.Group, { ...addDebugOutlineIfEnabled(), ref: ref, ...props, css: getCheckboxEmotionStyles(clsPrefix, theme, layout === 'horizontal', useNewShadows, useNewFormUISpacing, useNewBorderRadii), children: _jsx(RestoreAntDDefaultClsPrefix, { children: children }) }) }));
});
const CheckboxNamespace = /* #__PURE__ */ Object.assign(DuboisCheckbox, { Group: CheckboxGroup });
export const Checkbox = CheckboxNamespace;
// TODO: I'm doing this to support storybook's docgen;
// We should remove this once we have a better storybook integration,
// since these will be exposed in the library's exports.
export const __INTERNAL_DO_NOT_USE__Group = CheckboxGroup;
//# sourceMappingURL=Checkbox.js.map