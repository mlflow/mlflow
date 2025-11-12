import { jsxs as _jsxs, jsx as _jsx } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { useMergeRefs } from '@floating-ui/react';
import { Radio as AntDRadio } from 'antd';
import React, { createContext, forwardRef, useContext, useCallback, useRef, useImperativeHandle, useEffect, useMemo, } from 'react';
import { DesignSystemEventProviderAnalyticsEventTypes, DesignSystemEventProviderComponentTypes, useDesignSystemEventComponentCallbacks, } from '../DesignSystemEventProvider/DesignSystemEventProvider';
import { DesignSystemAntDConfigProvider, getAnimationCss } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks/useDesignSystemTheme';
import { useDesignSystemSafexFlags, useNotifyOnFirstView } from '../utils';
import { importantify } from '../utils/css-utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
import { safex } from '../utils/safex';
// TODO(GP): Add this to common spacing vars; I didn't want to make a decision on the value right now,
// so copied it from `Button`.
const SMALL_BUTTON_HEIGHT = 24;
function getSegmentedControlGroupEmotionStyles(clsPrefix, theme, spaced = false, truncateButtons, useSegmentedSliderStyle = false) {
    const classGroup = `.${clsPrefix}-radio-group`;
    const classSmallGroup = `.${clsPrefix}-radio-group-small`;
    const classButtonWrapper = `.${clsPrefix}-radio-button-wrapper`;
    const styles = {
        ...(truncateButtons && {
            display: 'flex',
            maxWidth: '100%',
        }),
        [`&${classGroup}`]: spaced ? { display: 'flex', gap: 8, flexWrap: 'wrap' } : {},
        [`&${classSmallGroup} ${classButtonWrapper}`]: {
            padding: '0 12px',
        },
    };
    const sliderStyles = {
        height: 'min-content',
        width: 'min-content',
        background: theme.colors.backgroundSecondary,
        borderRadius: theme.borders.borderRadiusMd,
        display: 'flex',
        gap: theme.spacing.xs,
    };
    const importantStyles = importantify(useSegmentedSliderStyle ? sliderStyles : styles);
    return css(importantStyles);
}
function getSegmentedControlButtonEmotionStyles(clsPrefix, theme, size, spaced = false, truncateButtons, useNewShadows, onlyIcon, useSegmentedSliderStyle = false) {
    const classWrapperChecked = `.${clsPrefix}-radio-button-wrapper-checked`;
    const classWrapper = `.${clsPrefix}-radio-button-wrapper`;
    const classWrapperDisabled = `.${clsPrefix}-radio-button-wrapper-disabled`;
    const classButton = `.${clsPrefix}-radio-button`;
    // Note: Ant radio button uses a 1px-wide `before` pseudo-element to recreate the left border of the button.
    // This is because the actual left border is disabled to avoid a double-border effect with the adjacent button's
    // right border.
    // We must override the background colour of this pseudo-border to be the same as the real border above.
    const styles = {
        backgroundColor: theme.colors.actionDefaultBackgroundDefault,
        borderColor: theme.colors.actionDefaultBorderDefault,
        color: theme.colors.actionDefaultTextDefault,
        ...(useNewShadows && {
            boxShadow: theme.shadows.xs,
        }),
        // This handles the left border of the button when they're adjacent
        '::before': {
            display: spaced ? 'none' : 'block',
            backgroundColor: theme.colors.actionDefaultBorderDefault,
        },
        '&:hover': {
            backgroundColor: theme.colors.actionDefaultBackgroundHover,
            borderColor: theme.colors.actionDefaultBorderHover,
            color: theme.colors.actionDefaultTextHover,
            '::before': {
                backgroundColor: theme.colors.actionDefaultBorderHover,
            },
            // Also target the same pseudo-element on the next sibling, because this is used to create the right border
            [`& + ${classWrapper}::before`]: {
                backgroundColor: theme.colors.actionDefaultBorderPress,
            },
        },
        '&:active': {
            backgroundColor: theme.colors.actionTertiaryBackgroundPress,
            borderColor: theme.colors.actionDefaultBorderPress,
            color: theme.colors.actionTertiaryTextPress,
        },
        [`&${classWrapperChecked}`]: {
            backgroundColor: theme.colors.actionTertiaryBackgroundPress,
            borderColor: theme.colors.actionDefaultBorderPress,
            color: theme.colors.actionTertiaryTextPress,
            ...(!useNewShadows && {
                boxShadow: 'none',
            }),
            '::before': {
                backgroundColor: theme.colors.actionDefaultBorderPress,
            },
            [`& + ${classWrapper}::before`]: {
                backgroundColor: theme.colors.actionDefaultBorderPress,
            },
        },
        [`&${classWrapperChecked}:focus-within`]: {
            '::before': {
                width: 0,
            },
        },
        [`&${classWrapper}`]: {
            padding: size === 'middle' ? `0 16px` : '0 8px',
            display: 'inline-flex',
            ...(onlyIcon
                ? {
                    '& > span': {
                        display: 'inline-flex',
                    },
                }
                : {}),
            verticalAlign: 'middle',
            ...(truncateButtons && {
                flexShrink: 1,
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
                minWidth: onlyIcon ? undefined : size === 'small' ? 58 : 68, // Don't allow the button to shrink and truncate below 3 characters
            }),
            '&:first-of-type': {
                borderTopLeftRadius: theme.borders.borderRadiusSm,
                borderBottomLeftRadius: theme.borders.borderRadiusSm,
            },
            '&:last-of-type': {
                borderTopRightRadius: theme.borders.borderRadiusSm,
                borderBottomRightRadius: theme.borders.borderRadiusSm,
            },
            ...(spaced
                ? {
                    borderWidth: 1,
                    borderRadius: theme.borders.borderRadiusSm,
                }
                : {}),
            '&:focus-within': {
                outlineStyle: 'solid',
                outlineWidth: '2px',
                outlineOffset: '-2px',
                outlineColor: theme.colors.actionDefaultBorderFocus,
            },
            ...(truncateButtons && {
                'span:last-of-type': {
                    textOverflow: 'ellipsis',
                    overflow: 'hidden',
                    whiteSpace: 'nowrap',
                },
            }),
        },
        [`&${classWrapper}, ${classButton}`]: {
            height: size === 'middle' ? theme.general.heightSm : SMALL_BUTTON_HEIGHT,
            lineHeight: theme.typography.lineHeightBase,
            alignItems: 'center',
        },
        [`&${classWrapperDisabled}, &${classWrapperDisabled} + ${classWrapperDisabled}`]: {
            color: theme.colors.actionDisabledText,
            backgroundColor: 'transparent',
            borderColor: theme.colors.actionDisabledBorder,
            '&:hover': {
                color: theme.colors.actionDisabledText,
                borderColor: theme.colors.actionDisabledBorder,
                backgroundColor: 'transparent',
            },
            '&:active': {
                color: theme.colors.actionDisabledText,
                borderColor: theme.colors.actionDisabledBorder,
                backgroundColor: 'transparent',
            },
            '::before': {
                backgroundColor: theme.colors.actionDisabledBorder,
            },
            [`&${classWrapperChecked}`]: {
                borderColor: theme.colors.actionDefaultBorderPress,
                '::before': {
                    backgroundColor: theme.colors.actionDefaultBorderPress,
                },
            },
            [`&${classWrapperChecked} + ${classWrapper}`]: {
                '::before': {
                    backgroundColor: theme.colors.actionDefaultBorderPress,
                },
            },
        },
        ...getAnimationCss(theme.options.enableAnimation),
    };
    const sliderStyles = {
        minWidth: 0,
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        whiteSpace: 'nowrap',
        border: 'none',
        display: 'inline-flex',
        alignItems: 'center',
        color: theme.colors.textSecondary,
        borderRadius: theme.borders.borderRadiusSm,
        backgroundColor: 'transparent',
        '::before': { display: 'none' },
        '&:hover': {
            backgroundColor: theme.colors.tableRowHover,
        },
        [`&${classWrapper}`]: {
            padding: size === 'middle' ? (onlyIcon ? `0 ${theme.spacing.sm}px` : `0 12px`) : `0 ${theme.spacing.sm}px`,
            verticalAlign: 'middle',
            ...(truncateButtons && {
                flexShrink: 1,
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
                minWidth: onlyIcon ? undefined : size === 'small' ? 58 : 68,
            }),
            '& > span': {
                display: 'inline-flex',
            },
        },
        [`&${classWrapper}, ${classButton}`]: {
            height: size === 'middle' ? theme.general.heightSm : SMALL_BUTTON_HEIGHT,
            lineHeight: theme.typography.lineHeightBase,
            alignItems: 'center',
        },
        [`&${classWrapperChecked}`]: {
            color: theme.colors.actionDefaultTextDefault,
            backgroundColor: theme.colors.backgroundPrimary,
            boxShadow: `inset 0 0 0 1px ${theme.colors.border}, ${theme.shadows.sm}`,
        },
        [`&${classWrapperDisabled}, &${classWrapperDisabled} + ${classWrapperDisabled}`]: {
            color: theme.colors.actionDisabledText,
            backgroundColor: 'transparent',
        },
        ...getAnimationCss(theme.options.enableAnimation),
    };
    const importantStyles = importantify(useSegmentedSliderStyle ? sliderStyles : styles);
    return css(importantStyles);
}
const SegmentedControlGroupContext = createContext({
    size: 'middle',
    spaced: false,
    dontTruncate: false,
    useSegmentedSliderStyle: false,
});
export const SegmentedControlButton = forwardRef(function SegmentedControlButton({ dangerouslySetAntdProps, ...props }, ref) {
    const { classNamePrefix, theme } = useDesignSystemTheme();
    const { size, spaced, dontTruncate, useSegmentedSliderStyle } = useContext(SegmentedControlGroupContext);
    const truncateButtons = safex('databricks.fe.designsystem.truncateSegmentedControlText', false) && !dontTruncate;
    const { useNewShadows } = useDesignSystemSafexFlags();
    const buttonRef = useRef(null);
    useImperativeHandle(ref, () => buttonRef.current);
    const onlyIcon = Boolean(props.icon && !props.children);
    const getLabelFromChildren = useCallback(() => {
        let label = '';
        React.Children.map(props.children, (child) => {
            if (typeof child === 'string') {
                label += child;
            }
        });
        return label;
    }, [props.children]);
    useEffect(() => {
        if (buttonRef.current) {
            // Using `as any` because Antd uses a `Checkbox` type that's not exported
            const labelParent = buttonRef.current.input.closest('label');
            if (labelParent) {
                labelParent.setAttribute('title', getLabelFromChildren());
            }
        }
    }, [buttonRef, getLabelFromChildren]);
    return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(AntDRadio.Button, { css: getSegmentedControlButtonEmotionStyles(classNamePrefix, theme, size, spaced, truncateButtons, useNewShadows, onlyIcon, useSegmentedSliderStyle), ...props, ...dangerouslySetAntdProps, ref: buttonRef, children: props.icon ? (_jsxs("div", { css: { display: 'inline-flex', alignItems: 'center', gap: theme.spacing.xs }, children: [props.icon, props.children] })) : (props.children) }) }));
});
export const SegmentedControlGroup = forwardRef(function SegmentedControlGroup({ dangerouslySetAntdProps, size = 'middle', spaced = false, onChange, componentId, analyticsEvents, valueHasNoPii, dontTruncate, newStyleFlagOverride, ...props }, ref) {
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.segmentedControlGroup', false);
    const { classNamePrefix, theme } = useDesignSystemTheme();
    const truncateButtons = safex('databricks.fe.designsystem.truncateSegmentedControlText', false) && !dontTruncate;
    const useSegmentedSliderStyle = newStyleFlagOverride ?? safex('databricks.fe.designsystem.useNewSegmentedControlStyles', false);
    const memoizedAnalyticsEvents = useMemo(() => analyticsEvents ??
        (emitOnView
            ? [
                DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
                DesignSystemEventProviderAnalyticsEventTypes.OnView,
            ]
            : [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange]), [analyticsEvents, emitOnView]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.SegmentedControlGroup,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        valueHasNoPii,
    });
    const { elementRef: segmentedControlGroupRef } = useNotifyOnFirstView({
        onView: eventContext.onView,
        value: props.value ?? props.defaultValue,
    });
    const mergedRef = useMergeRefs([ref, segmentedControlGroupRef]);
    const onChangeWrapper = useCallback((e) => {
        eventContext.onValueChange(e.target.value);
        onChange?.(e);
    }, [eventContext, onChange]);
    return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(SegmentedControlGroupContext.Provider, { value: { size, spaced, dontTruncate, useSegmentedSliderStyle }, children: _jsx(AntDRadio.Group, { ...addDebugOutlineIfEnabled(), ...props, css: getSegmentedControlGroupEmotionStyles(classNamePrefix, theme, spaced, truncateButtons, useSegmentedSliderStyle), onChange: onChangeWrapper, ...dangerouslySetAntdProps, ref: mergedRef, ...eventContext.dataComponentProps }) }) }));
});
//# sourceMappingURL=SegmentedControl.js.map