import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { useMergeRefs } from '@floating-ui/react';
import { Radio as AntDRadio } from 'antd';
import React, { forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef } from 'react';
import { DesignSystemEventProviderAnalyticsEventTypes, DesignSystemEventProviderComponentTypes, useDesignSystemEventComponentCallbacks, } from '../DesignSystemEventProvider/DesignSystemEventProvider';
import { DesignSystemAntDConfigProvider, getAnimationCss, RestoreAntDDefaultClsPrefix } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks';
import { useDesignSystemSafexFlags, useNotifyOnFirstView } from '../utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
import { safex } from '../utils/safex';
const RadioGroupContext = React.createContext(undefined);
export const useRadioGroupContext = () => {
    const context = React.useContext(RadioGroupContext);
    if (!context) {
        throw new Error('Radio components are only allowed within a Radio.Group');
    }
    return context;
};
const getRadioInputStyles = ({ clsPrefix, theme, useNewShadows, }) => ({
    [`.${clsPrefix}`]: {
        alignSelf: 'start',
        // Unchecked Styles
        [`> .${clsPrefix}-input + .${clsPrefix}-inner`]: {
            width: theme.spacing.md,
            height: theme.spacing.md,
            background: theme.colors.actionDefaultBackgroundDefault,
            borderStyle: 'solid',
            borderColor: theme.colors.actionDefaultBorderDefault,
            boxShadow: 'unset',
            transform: 'unset', // This prevents an awkward jitter on the border
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            borderRadius: theme.borders.borderRadiusFull,
            '&:after': {
                all: 'unset',
            },
        },
        // Hover
        [`&:not(.${clsPrefix}-disabled) > .${clsPrefix}-input:hover + .${clsPrefix}-inner`]: {
            borderColor: theme.colors.actionPrimaryBackgroundHover,
            background: theme.colors.actionDefaultBackgroundHover,
        },
        // Focus
        [`&:not(.${clsPrefix}-disabled)> .${clsPrefix}-input:focus + .${clsPrefix}-inner`]: {
            borderColor: theme.colors.actionPrimaryBackgroundDefault,
        },
        // Active
        [`&:not(.${clsPrefix}-disabled)> .${clsPrefix}-input:active + .${clsPrefix}-inner`]: {
            borderColor: theme.colors.actionPrimaryBackgroundPress,
            background: theme.colors.actionDefaultBackgroundPress,
        },
        // Disabled
        [`&.${clsPrefix}-disabled > .${clsPrefix}-input + .${clsPrefix}-inner`]: {
            borderColor: `${theme.colors.actionDisabledBorder} !important`, // Ant uses !important
            background: theme.colors.actionDisabledBackground,
            '@media (forced-colors: active)': {
                borderColor: 'GrayText !important',
            },
        },
        // Checked Styles
        [`&.${clsPrefix}-checked`]: {
            '&:after': {
                border: 'unset',
            },
            [`> .${clsPrefix}-input + .${clsPrefix}-inner`]: {
                background: theme.colors.actionPrimaryBackgroundDefault,
                borderColor: theme.colors.primary,
                ...(useNewShadows && {
                    boxShadow: theme.shadows.xs,
                }),
                '&:after': {
                    content: `""`,
                    borderRadius: theme.spacing.xs,
                    backgroundColor: theme.colors.white,
                    width: theme.spacing.xs,
                    height: theme.spacing.xs,
                },
                '@media (forced-colors: active)': {
                    borderColor: 'Highlight !important',
                    backgroundColor: 'Highlight !important',
                },
            },
            // Hover
            [`&:not(.${clsPrefix}-disabled) > .${clsPrefix}-input:hover + .${clsPrefix}-inner`]: {
                background: theme.colors.actionPrimaryBackgroundHover,
                borderColor: theme.colors.actionPrimaryBackgroundPress,
            },
            // Focus
            [`&:not(.${clsPrefix}-disabled) > .${clsPrefix}-input:focus-visible + .${clsPrefix}-inner`]: {
                background: theme.colors.actionPrimaryBackgroundDefault,
                borderColor: theme.colors.actionDefaultBorderFocus,
                outline: `1px solid ${theme.colors.actionPrimaryBackgroundDefault}`,
                outlineOffset: 1,
            },
            // Active
            [`&:not(.${clsPrefix}-disabled) > .${clsPrefix}-input:active + .${clsPrefix}-inner`]: {
                background: theme.colors.actionDefaultBackgroundPress,
                borderColor: theme.colors.actionDefaultBorderPress,
            },
            // Disabled
            [`&.${clsPrefix}-disabled > .${clsPrefix}-input + .${clsPrefix}-inner`]: {
                background: theme.colors.actionDisabledBackground,
                border: `1px solid ${theme.colors.actionDisabledBorder} !important`,
                '&:after': {
                    backgroundColor: theme.colors.actionDisabledText,
                },
                '@media (forced-colors: active)': {
                    borderColor: 'GrayText !important',
                    backgroundColor: 'GrayText !important',
                },
            },
        },
    },
});
const getCommonRadioGroupStyles = ({ theme, clsPrefix, classNamePrefix, useNewShadows, }) => css({
    '& > label': {
        [`&.${classNamePrefix}-radio-wrapper-disabled > span`]: {
            color: theme.colors.actionDisabledText,
        },
    },
    [`& > label + .${classNamePrefix}-hint`]: {
        paddingLeft: theme.spacing.lg,
    },
    ...getRadioInputStyles({ theme, clsPrefix, useNewShadows }),
    ...getAnimationCss(theme.options.enableAnimation),
});
const getHorizontalRadioGroupStyles = ({ theme, classNamePrefix, useEqualColumnWidths, useNewFormUISpacing, }) => css({
    '&&': {
        display: 'grid',
        gridTemplateRows: '[label] auto [hint] auto',
        gridAutoColumns: useEqualColumnWidths ? 'minmax(0, 1fr)' : 'max-content',
        gridColumnGap: theme.spacing.md,
        ...(useNewFormUISpacing && {
            [`& + .${classNamePrefix}-form-message`]: {
                marginTop: theme.spacing.sm,
            },
        }),
    },
    ...(useNewFormUISpacing && {
        [`:has(> .${classNamePrefix}-hint)`]: {
            marginTop: theme.spacing.sm,
        },
    }),
    [`& > label, & > .${classNamePrefix}-radio-tile`]: {
        gridRow: 'label / label',
        marginRight: 0,
    },
    [`& > label + .${classNamePrefix}-hint`]: {
        display: 'inline-block',
        gridRow: 'hint / hint',
    },
});
const getVerticalRadioGroupStyles = ({ theme, classNamePrefix, useNewFormUISpacing, }) => css({
    display: 'flex',
    flexDirection: 'column',
    flexWrap: 'wrap',
    ...(useNewFormUISpacing && {
        [`& + .${classNamePrefix}-form-message`]: {
            marginTop: theme.spacing.sm,
        },
        [`~ .${classNamePrefix}-label)`]: {
            marginTop: theme.spacing.sm,
            background: 'red',
        },
    }),
    [`.${classNamePrefix}-radio-tile + .${classNamePrefix}-radio-tile`]: {
        marginTop: theme.spacing.md,
    },
    '& > label': {
        fontWeight: 'normal',
        paddingBottom: theme.spacing.sm,
    },
    [`& > label:last-of-type`]: {
        paddingBottom: 0,
    },
    [`& > label + .${classNamePrefix}-hint`]: {
        marginBottom: theme.spacing.sm,
        paddingLeft: theme.spacing.lg,
        marginTop: `-${theme.spacing.sm}px`,
    },
    [`& > label:last-of-type + .${classNamePrefix}-hint`]: {
        marginTop: 0,
    },
});
export const getRadioStyles = ({ theme, clsPrefix, useNewShadows, }) => {
    // Default as bold for standalone radios
    const fontWeight = 'normal';
    const styles = {
        fontWeight,
    };
    return css({ ...getRadioInputStyles({ theme, clsPrefix, useNewShadows }), ...styles });
};
const DuboisRadio = forwardRef(function Radio({ children, dangerouslySetAntdProps, __INTERNAL_DISABLE_RADIO_ROLE, componentId, analyticsEvents, valueHasNoPii, onChange, ...props }, ref) {
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.radio', false);
    const { theme, getPrefixedClassName } = useDesignSystemTheme();
    const { componentId: contextualComponentId } = React.useContext(RadioGroupContext) ?? {};
    const { useNewShadows } = useDesignSystemSafexFlags();
    const clsPrefix = getPrefixedClassName('radio');
    const memoizedAnalyticsEvents = useMemo(() => analyticsEvents ??
        (emitOnView
            ? [
                DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
                DesignSystemEventProviderAnalyticsEventTypes.OnView,
            ]
            : [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange]), [analyticsEvents, emitOnView]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.Radio,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        valueHasNoPii,
    });
    const firstView = useRef(true);
    useEffect(() => {
        // Only call the onView callback if the Radio is standalone and not part of a RadioGroup
        if (componentId && contextualComponentId === undefined && firstView.current) {
            eventContext.onView(props.value);
            firstView.current = false;
        }
    }, [eventContext, componentId, contextualComponentId, props.value]);
    const onChangeWrapper = useCallback((e) => {
        // Only call the onValueChange callback if the Radio is standalone and not part of a RadioGroup
        if (contextualComponentId === undefined) {
            eventContext.onValueChange?.(e.target.value);
        }
        onChange?.(e);
    }, [contextualComponentId, eventContext, onChange]);
    return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(AntDRadio, { ...addDebugOutlineIfEnabled(), css: getRadioStyles({ theme, clsPrefix, useNewShadows }), ...props, ...dangerouslySetAntdProps, ...(__INTERNAL_DISABLE_RADIO_ROLE ? { role: 'none' } : {}), onChange: onChangeWrapper, ref: ref, "data-component-type": contextualComponentId
                ? DesignSystemEventProviderComponentTypes.RadioGroup
                : DesignSystemEventProviderComponentTypes.Radio, "data-component-id": contextualComponentId ?? componentId, children: _jsx(RestoreAntDDefaultClsPrefix, { children: children }) }) }));
});
const StyledRadioGroup = forwardRef(function StyledRadioGroup({ children, dangerouslySetAntdProps, componentId, analyticsEvents, valueHasNoPii, onChange, ...props }, ref) {
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.radio', false);
    const { theme, getPrefixedClassName, classNamePrefix } = useDesignSystemTheme();
    const { useNewShadows } = useDesignSystemSafexFlags();
    const clsPrefix = getPrefixedClassName('radio');
    const memoizedAnalyticsEvents = useMemo(() => analyticsEvents ??
        (emitOnView
            ? [
                DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
                DesignSystemEventProviderAnalyticsEventTypes.OnView,
            ]
            : [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange]), [analyticsEvents, emitOnView]);
    const [value, setValue] = React.useState(props.defaultValue ?? '');
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.RadioGroup,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        valueHasNoPii,
    });
    const internalRef = useRef();
    useImperativeHandle(ref, () => internalRef.current);
    const { elementRef: radioGroupRef } = useNotifyOnFirstView({
        onView: eventContext.onView,
        value: props.value ?? props.defaultValue,
    });
    const mergedRef = useMergeRefs([internalRef, radioGroupRef]);
    const onChangeWrapper = useCallback((e) => {
        eventContext.onValueChange?.(e.target.value);
        setValue(e.target.value);
        onChange?.(e);
    }, [eventContext, onChange]);
    useEffect(() => {
        if (props.value !== undefined) {
            setValue(props.value);
            // Antd's Radio (rc-checkbox) is not updating checked state correctly even though state is managed appropriately on our end
            // Manually add and remove checked attribute to the radio input to ensure it is checked and A11y tools and tests can rely on this for validation
            if (internalRef?.current) {
                // Remove checked attribute from old radio input
                const checkedInput = internalRef.current.querySelector('input[checked]');
                if (checkedInput) {
                    checkedInput.removeAttribute('checked');
                }
                // Add checked attribute to new radio input
                const toBeCheckedInput = internalRef.current.querySelector(`input[value="${props.value}"]`);
                if (toBeCheckedInput) {
                    toBeCheckedInput.setAttribute('checked', 'checked');
                }
            }
        }
    }, [props.value]);
    return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(RadioGroupContext.Provider, { value: { componentId, value, onChange: onChangeWrapper }, children: _jsx(AntDRadio.Group, { ...addDebugOutlineIfEnabled(), ...props, css: getCommonRadioGroupStyles({
                    theme,
                    clsPrefix,
                    classNamePrefix,
                    useNewShadows,
                }), value: value, onChange: onChangeWrapper, ...dangerouslySetAntdProps, ref: mergedRef, children: _jsx(RestoreAntDDefaultClsPrefix, { children: children }) }) }) }));
});
const HorizontalGroup = forwardRef(function HorizontalGroup({ dangerouslySetAntdProps, useEqualColumnWidths, ...props }, ref) {
    const { theme, classNamePrefix } = useDesignSystemTheme();
    const { useNewFormUISpacing } = useDesignSystemSafexFlags();
    return (_jsx(StyledRadioGroup, { css: getHorizontalRadioGroupStyles({ theme, classNamePrefix, useEqualColumnWidths, useNewFormUISpacing }), ...props, ref: ref, ...dangerouslySetAntdProps }));
});
const Group = forwardRef(function HorizontalGroup({ dangerouslySetAntdProps, layout = 'vertical', useEqualColumnWidths, ...props }, ref) {
    const { theme, classNamePrefix } = useDesignSystemTheme();
    const { useNewFormUISpacing } = useDesignSystemSafexFlags();
    return (_jsx(StyledRadioGroup, { css: layout === 'horizontal'
            ? getHorizontalRadioGroupStyles({ theme, classNamePrefix, useEqualColumnWidths, useNewFormUISpacing })
            : getVerticalRadioGroupStyles({
                theme,
                classNamePrefix,
                useNewFormUISpacing,
            }), ...props, ref: ref, ...dangerouslySetAntdProps }));
});
// Note: We are overriding ant's default "Group" with our own.
const RadioNamespace = /* #__PURE__ */ Object.assign(DuboisRadio, { Group, HorizontalGroup });
export const Radio = RadioNamespace;
// TODO: I'm doing this to support storybook's docgen;
// We should remove this once we have a better storybook integration,
// since these will be exposed in the library's exports.
// We should ideally be using __Group instead of __VerticalGroup, but that exists under Checkbox too and conflicts, therefore
// we show a wrong component name in "Show code" in docs, fix included in story to replace this with correct name
export const __INTERNAL_DO_NOT_USE__VerticalGroup = Group;
export const __INTERNAL_DO_NOT_USE__HorizontalGroup = HorizontalGroup;
//# sourceMappingURL=Radio.js.map