import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { Button as AntDButton } from 'antd';
import { forwardRef, useCallback, useEffect, useImperativeHandle, useMemo } from 'react';
import { getDefaultStyles, getDisabledDefaultStyles, getDisabledErrorStyles, getDisabledPrimaryStyles, getDisabledTertiaryStyles, getLinkStyles, getPrimaryDangerStyles, getPrimaryStyles, getSecondaryDangerStyles, } from './styles';
import { useFormContext } from '../../development/Form/Form';
import { DesignSystemEventProviderAnalyticsEventTypes, DesignSystemEventProviderComponentTypes, useDesignSystemEventComponentCallbacks, } from '../DesignSystemEventProvider';
import { DesignSystemAntDConfigProvider, getAnimationCss } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks';
import { Spinner } from '../Spinner';
import { useDesignSystemSafexFlags, useNotifyOnFirstView } from '../utils';
import { importantify } from '../utils/css-utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
import { safex } from '../utils/safex';
const buttonStylesCache = new WeakMap();
export const getMemoizedButtonEmotionStyles = (props) => {
    // Theme is a large object, so cache it by ref using a WeakMap,
    // and then JSON.stringify() the rest of the props as the cache key.
    const { theme, ...rest } = props;
    const themeCache = buttonStylesCache.get(theme) || new Map();
    if (!buttonStylesCache.has(theme)) {
        buttonStylesCache.set(theme, themeCache);
    }
    const cacheKey = JSON.stringify(rest);
    const cachedStyles = themeCache.get(cacheKey);
    if (cachedStyles) {
        return cachedStyles;
    }
    const styles = getButtonEmotionStyles(props);
    themeCache.set(cacheKey, styles);
    return styles;
};
function getEndIconClsName(theme) {
    return `${theme.general.iconfontCssPrefix}-btn-end-icon`;
}
export const getButtonEmotionStyles = ({ theme, classNamePrefix, loading, withIcon, onlyIcon, isAnchor, enableAnimation, size, type, useFocusPseudoClass, forceIconStyles, danger, useNewShadows, useNewBorderRadii, }) => {
    const clsIcon = `.${theme.general.iconfontCssPrefix}`;
    const clsEndIcon = `.${getEndIconClsName(theme)}`;
    const clsLoadingIcon = `.${classNamePrefix}-btn-loading-icon`;
    const clsIconOnly = `.${classNamePrefix}-btn-icon-only`;
    const classPrimary = `.${classNamePrefix}-btn-primary`;
    const classLink = `.${classNamePrefix}-btn-link`;
    const classDangerous = `.${classNamePrefix}-btn-dangerous`;
    const SMALL_BUTTON_HEIGHT = 24;
    const tertiaryColors = {
        background: theme.colors.actionTertiaryBackgroundDefault,
        color: theme.colors.actionTertiaryTextDefault,
        ...(useNewShadows && {
            boxShadow: 'none',
        }),
        '&:hover': {
            background: theme.colors.actionTertiaryBackgroundHover,
            color: theme.colors.actionTertiaryTextHover,
        },
        '&:active': {
            background: theme.colors.actionTertiaryBackgroundPress,
            color: theme.colors.actionTertiaryTextPress,
        },
    };
    const iconCss = {
        fontSize: theme.general.iconFontSize,
        lineHeight: 0,
        ...(size === 'small' && {
            lineHeight: theme.typography.lineHeightSm,
            height: 16,
            ...((onlyIcon || forceIconStyles) && {
                fontSize: 16,
            }),
        }),
    };
    const inactiveIconCss = {
        color: theme.colors.textSecondary,
    };
    const endIconCssSelector = `span > ${clsEndIcon} > ${clsIcon}`;
    const styles = {
        lineHeight: theme.typography.lineHeightBase,
        boxShadow: useNewShadows ? theme.shadows.xs : 'none',
        height: theme.general.heightSm,
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        verticalAlign: 'middle',
        ...(useNewBorderRadii && {
            borderRadius: theme.borders.borderRadiusSm,
        }),
        ...(!onlyIcon &&
            !forceIconStyles && {
            '&&': {
                padding: '4px 12px',
                ...(size === 'small' && {
                    padding: '0 8px',
                }),
            },
        }),
        ...((onlyIcon || forceIconStyles) && {
            width: theme.general.heightSm,
        }),
        ...(size === 'small' && {
            height: SMALL_BUTTON_HEIGHT,
            lineHeight: theme.typography.lineHeightBase,
            ...((onlyIcon || forceIconStyles) && {
                width: SMALL_BUTTON_HEIGHT,
                paddingTop: 0,
                paddingBottom: 0,
                verticalAlign: 'middle',
            }),
        }),
        '&:focus-visible': {
            outlineStyle: 'solid',
            outlineWidth: '2px',
            outlineOffset: '1px',
            outlineColor: theme.isDarkMode ? theme.colors.actionDefaultBorderFocus : theme.colors.actionDefaultBorderFocus,
        },
        ...getDefaultStyles(theme, loading),
        [`&${classPrimary}`]: {
            ...getPrimaryStyles(theme),
        },
        [`&${classLink}`]: {
            ...getLinkStyles(theme),
            ...(type === 'link' && {
                padding: 'unset',
                height: 'auto',
                border: 'none',
                boxShadow: 'none',
                '&[disabled],&:hover': {
                    background: 'none',
                },
            }),
        },
        [`&${classDangerous}${classPrimary}`]: {
            ...getPrimaryDangerStyles(theme),
        },
        [`&${classDangerous}`]: {
            ...getSecondaryDangerStyles(theme),
        },
        [`&[disabled]`]: {
            ...getDisabledDefaultStyles(theme, useNewShadows),
        },
        [`&${classLink}:disabled`]: {
            ...getDisabledTertiaryStyles(theme, useNewShadows),
        },
        [`&${classDangerous}:disabled`]: {
            ...getDisabledErrorStyles(theme, useNewShadows),
        },
        [`&${classPrimary}:disabled`]: {
            ...getDisabledPrimaryStyles(theme, useNewShadows),
        },
        [`&[disabled], &${classDangerous}:disabled`]: {
            ...(useNewShadows && { boxShadow: 'none' }),
            ...((onlyIcon || forceIconStyles) && {
                backgroundColor: 'transparent',
                '&:hover': {
                    backgroundColor: 'transparent',
                },
                '&:active': {
                    backgroundColor: 'transparent',
                },
            }),
        },
        [clsLoadingIcon]: {
            display: 'none',
        },
        // Loading styles
        ...(loading && {
            '::before': {
                opacity: 0,
            },
            cursor: 'default',
            [`${clsLoadingIcon}`]: {
                ...(onlyIcon
                    ? {
                        // In case of only icon, the icon is already centered but vertically not aligned, this fixes that
                        verticalAlign: 'middle',
                    }
                    : {
                        // Position loading indicator in center
                        // This would break vertical centering of loading circle when onlyIcon is true
                        position: 'absolute',
                    }),
                // Re-enable animation for the loading spinner, since it can be disabled by the global animation CSS.
                svg: {
                    animationDuration: '1s !important',
                },
            },
            [`& > ${clsLoadingIcon} .anticon`]: {
                paddingRight: 0, // to horizontally center icon
            },
            [`> :not(${clsLoadingIcon})`]: {
                // Hide all content except loading icon
                opacity: 0,
                visibility: 'hidden',
                // Add horizontal space for icon
                ...(withIcon && { paddingLeft: theme.general.iconFontSize + theme.spacing.xs }),
            },
        }),
        // Icon styles
        [`> ${clsIcon} + span, > span + ${clsIcon}`]: {
            marginRight: 0,
            marginLeft: theme.spacing.xs,
        },
        [`> ${clsIcon}`]: iconCss,
        [`> ${endIconCssSelector}`]: {
            ...iconCss,
            marginLeft: size === 'small' ? theme.spacing.xs : theme.spacing.sm,
        },
        ...(!type &&
            !danger && {
            [`&:enabled:not(:hover):not(:active) > ${clsIcon}`]: inactiveIconCss,
        }),
        ...(!type &&
            !danger && {
            [`&:enabled:not(:hover):not(:active) > ${endIconCssSelector}`]: inactiveIconCss,
        }),
        // Disable animations
        [`&[${classNamePrefix}-click-animating-without-extra-node='true']::after`]: {
            display: 'none',
        },
        [`&${clsIconOnly}`]: {
            border: 'none',
            ...(useNewShadows && {
                boxShadow: 'none',
            }),
            [`&:enabled:not(${classLink})`]: {
                ...tertiaryColors,
                color: theme.colors.textSecondary,
                '&:hover > .anticon': {
                    color: tertiaryColors['&:hover'].color,
                    ...(danger && {
                        color: theme.colors.actionDangerDefaultTextHover,
                    }),
                },
                '&:active > .anticon': {
                    color: tertiaryColors['&:active'].color,
                    ...(danger && {
                        color: theme.colors.actionDangerDefaultTextPress,
                    }),
                },
                ...(loading && {
                    '&&, &:hover, &:active': {
                        backgroundColor: 'transparent',
                    },
                }),
            },
            [`&:enabled:not(${classLink}) > .anticon`]: {
                color: theme.colors.textSecondary,
                ...(danger && {
                    color: theme.colors.actionDangerDefaultTextDefault,
                }),
            },
            ...(isAnchor && {
                lineHeight: `${theme.general.heightSm}px`,
                ...getLinkStyles(theme),
                '&:disabled': {
                    color: theme.colors.actionDisabledText,
                },
            }),
            ...(loading && {
                '&&, &:hover, &:active': {
                    backgroundColor: 'transparent',
                },
            }),
            '&[disabled]:hover': {
                backgroundColor: 'transparent',
            },
        },
        [`&:focus`]: {
            ...(useFocusPseudoClass && {
                outlineStyle: 'solid',
                outlineWidth: '2px',
                outlineOffset: '1px',
                outlineColor: theme.isDarkMode ? theme.colors.actionDefaultBorderFocus : theme.colors.actionDefaultBorderFocus,
            }),
            [`${clsLoadingIcon}`]: {
                ...(onlyIcon && {
                    // Mitigate wrong left offset for loading state with onlyIcon
                    left: 0,
                }),
            },
        },
        ...(forceIconStyles && {
            padding: '0 6px',
            lineHeight: theme.typography.lineHeightSm,
            color: theme.colors.textSecondary,
            ...(loading && {
                '&&, &:hover, &:active': {
                    backgroundColor: 'transparent',
                    borderColor: theme.colors.actionDefaultBorderDefault,
                },
                '&[disabled], &[disabled]:hover, &[disabled]:active': {
                    backgroundColor: 'transparent',
                    borderColor: 'transparent',
                },
            }),
            '& > span': {
                verticalAlign: -1,
                height: theme.general.heightSm / 2,
                width: theme.general.heightSm / 2,
            },
            [`& > ${clsLoadingIcon} .anticon`]: {
                // left: `calc(50% - 6px)!important`,
                height: theme.general.heightSm / 2,
                width: theme.general.heightSm / 2,
                padding: 0,
            },
        }),
        ...getAnimationCss(enableAnimation),
    };
    // Moved outside main style object because state & selector matching in the already existing object keys can create bugs and unwanted overwrites
    const typeStyles = {
        ...(type === 'tertiary' && {
            [`&:enabled:not(${clsIconOnly})`]: tertiaryColors,
            [`&${classLink}[disabled]`]: {
                ...getDisabledTertiaryStyles(theme, useNewShadows),
            },
        }),
    };
    const importantStyles = importantify(styles);
    const importantTypeStyles = importantify(typeStyles);
    return css(importantStyles, importantTypeStyles);
};
export const Button = /* #__PURE__ */ (() => {
    const Button = forwardRef(function Button(
    // Keep size out of props passed to AntD to make deprecation and eventual removal have 0 impact
    { dangerouslySetAntdProps, children, size, type, loading: loadingProp, loadingDescription, endIcon, onClick, dangerouslySetForceIconStyles, dangerouslyUseFocusPseudoClass, dangerouslyAppendWrapperCss, componentId, analyticsEvents, shouldStartInteraction, ...props }, ref) {
        const emitOnView = safex('databricks.fe.observability.defaultButtonComponentView', true);
        const formContext = useFormContext();
        const { theme, classNamePrefix } = useDesignSystemTheme();
        const { useNewShadows, useNewBorderRadii } = useDesignSystemSafexFlags();
        const memoizedAnalyticsEvents = useMemo(() => analyticsEvents ??
            (emitOnView
                ? [DesignSystemEventProviderAnalyticsEventTypes.OnClick, DesignSystemEventProviderAnalyticsEventTypes.OnView]
                : [DesignSystemEventProviderAnalyticsEventTypes.OnClick]), [analyticsEvents, emitOnView]);
        const eventContext = useDesignSystemEventComponentCallbacks({
            componentType: DesignSystemEventProviderComponentTypes.Button,
            componentId,
            analyticsEvents: memoizedAnalyticsEvents,
            shouldStartInteraction,
            // If the button is a submit button and is part of a form, it is not the subject of the interaction, the form submission is
            isInteractionSubject: !(props.htmlType === 'submit' && formContext.componentId),
        });
        const clsEndIcon = getEndIconClsName(theme);
        const loadingCls = `${classNamePrefix}-btn-loading-icon`;
        const { elementRef: buttonRef } = useNotifyOnFirstView({ onView: eventContext.onView });
        useImperativeHandle(ref, () => buttonRef.current);
        const loading = loadingProp ?? (props.htmlType === 'submit' && formContext.isSubmitting);
        // Needed to keep backwards compatibility and support existing unit tests
        useEffect(() => {
            if (buttonRef.current) {
                if (loading) {
                    buttonRef.current.setAttribute('loading', 'true');
                    buttonRef.current.classList.add(`${classNamePrefix}-btn-loading`);
                }
                else {
                    buttonRef.current.setAttribute('loading', 'false');
                    buttonRef.current.classList.remove(`${classNamePrefix}-btn-loading`);
                }
            }
        }, [loading, classNamePrefix, buttonRef]);
        const iconOnly = Boolean((props.icon || endIcon) && !children);
        const handleClick = useCallback((event) => {
            if (loading) {
                return;
            }
            eventContext.onClick(event);
            if (props.htmlType === 'submit' && formContext.formRef?.current) {
                event.preventDefault();
                formContext.formRef.current.requestSubmit();
            }
            onClick?.(event);
        }, [loading, props.htmlType, formContext.formRef, eventContext, onClick]);
        const loadingSpinner = (_jsx(Spinner, { className: loadingCls, animationDuration: 8, inheritColor: true, label: "loading", "aria-label": "loading", loadingDescription: loadingDescription ?? componentId, css: {
                color: 'inherit !important',
                pointerEvents: 'none',
                ...(!iconOnly &&
                    !dangerouslySetForceIconStyles && {
                    '.anticon': {
                        verticalAlign: '-0.2em',
                    },
                }),
                '[aria-hidden="true"]': {
                    display: 'inline',
                },
            } }));
        return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(AntDButton, { ...addDebugOutlineIfEnabled(), ...props, css: getMemoizedButtonEmotionStyles({
                    theme,
                    classNamePrefix,
                    loading: Boolean(loading),
                    withIcon: Boolean(props.icon),
                    onlyIcon: iconOnly,
                    isAnchor: Boolean(props.href && !type),
                    danger: Boolean(props.danger),
                    enableAnimation: theme.options.enableAnimation,
                    size: size || 'middle',
                    type,
                    forceIconStyles: Boolean(dangerouslySetForceIconStyles),
                    useFocusPseudoClass: Boolean(dangerouslyUseFocusPseudoClass),
                    useNewShadows,
                    useNewBorderRadii,
                }), href: props.disabled ? undefined : props.href, ...dangerouslySetAntdProps, onClick: handleClick, icon: loading ? loadingSpinner : props.icon, ref: buttonRef, type: type === 'tertiary' ? 'link' : type, ...eventContext.dataComponentProps, children: children && (_jsxs("span", { style: {
                        visibility: loading ? 'hidden' : undefined,
                        display: 'inline-flex',
                        alignItems: 'center',
                        ...dangerouslyAppendWrapperCss,
                    }, children: [children, endIcon && (_jsx("span", { className: clsEndIcon, style: { display: 'inline-flex', alignItems: 'center' }, children: endIcon }))] })) }) }));
    });
    // This is needed for other Ant components that wrap Button, such as Tooltip, to correctly
    // identify it as an Ant button.
    // This should be removed if the component is rewritten to no longer be a wrapper around Ant.
    // See: https://github.com/ant-design/ant-design/blob/6dd39c1f89b4d6632e6ed022ff1bc275ca1e0f1f/components/button/button.tsx#L291
    Button.__ANT_BUTTON = true;
    return Button;
})();
//# sourceMappingURL=Button.js.map