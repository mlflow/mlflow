import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { useMergeRefs } from '@floating-ui/react';
import { useCallback, forwardRef, useMemo } from 'react';
import { DesignSystemEventProviderAnalyticsEventTypes, DesignSystemEventProviderComponentTypes, useDesignSystemEventComponentCallbacks, } from '../DesignSystemEventProvider';
import { useDesignSystemTheme } from '../Hooks/useDesignSystemTheme';
import { CloseIcon } from '../Icon';
import { useNotifyOnFirstView } from '../utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
import { safex } from '../utils/safex';
const oldTagColorsMap = {
    default: 'tagDefault',
    brown: 'tagBrown',
    coral: 'tagCoral',
    charcoal: 'grey600',
    indigo: 'tagIndigo',
    lemon: 'tagLemon',
    lime: 'tagLime',
    pink: 'tagPink',
    purple: 'tagPurple',
    teal: 'tagTeal',
    turquoise: 'tagTurquoise',
};
function getTagEmotionStyles(theme, color = 'default', clickable = false, closable = false) {
    let textColor = theme.colors.tagText;
    let backgroundColor = theme.colors[oldTagColorsMap[color]];
    let iconColor = '';
    let outlineColor = theme.colors.actionDefaultBorderFocus;
    const capitalizedColor = (color.charAt(0).toUpperCase() + color.slice(1));
    textColor = theme.DU_BOIS_INTERNAL_ONLY.colors[`tagText${capitalizedColor}`];
    backgroundColor = theme.DU_BOIS_INTERNAL_ONLY.colors[`tagBackground${capitalizedColor}`];
    iconColor = theme.DU_BOIS_INTERNAL_ONLY.colors[`tagIcon${capitalizedColor}`];
    if (color === 'charcoal') {
        outlineColor = theme.colors.white;
    }
    const iconHover = theme.colors.tagIconHover;
    const iconPress = theme.colors.tagIconPress;
    return {
        wrapper: {
            backgroundColor: backgroundColor,
            display: 'inline-flex',
            alignItems: 'center',
            marginRight: theme.spacing.sm,
            borderRadius: theme.borders.borderRadiusSm,
        },
        tag: {
            border: 'none',
            color: textColor,
            padding: '',
            backgroundColor: 'transparent',
            borderRadius: theme.borders.borderRadiusSm,
            marginRight: theme.spacing.sm,
            display: 'inline-block',
            cursor: clickable ? 'pointer' : 'default',
            ...(closable && {
                borderTopRightRadius: 0,
                borderBottomRightRadius: 0,
            }),
            ...(clickable && {
                '&:hover': {
                    '& > div': {
                        backgroundColor: theme.colors.actionDefaultBackgroundHover,
                    },
                },
                '&:active': {
                    '& > div': {
                        backgroundColor: theme.colors.actionDefaultBackgroundPress,
                    },
                },
            }),
        },
        content: {
            display: 'flex',
            alignItems: 'center',
            minWidth: 0,
            height: theme.typography.lineHeightBase,
        },
        close: {
            height: theme.typography.lineHeightBase,
            width: theme.typography.lineHeightBase,
            lineHeight: `${theme.general.iconFontSize}px`,
            padding: 0,
            color: textColor,
            fontSize: theme.general.iconFontSize,
            borderTopRightRadius: theme.borders.borderRadiusSm,
            borderBottomRightRadius: theme.borders.borderRadiusSm,
            border: 'none',
            background: 'none',
            cursor: 'pointer',
            marginLeft: theme.spacing.xs,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: 0,
            '&:hover': {
                backgroundColor: theme.colors.actionDefaultBackgroundHover,
                color: iconHover,
            },
            '&:active': {
                backgroundColor: theme.colors.actionDefaultBackgroundPress,
                color: iconPress,
            },
            '&:focus-visible': {
                outlineStyle: 'solid',
                outlineWidth: 1,
                outlineOffset: -2,
                outlineColor,
            },
            '.anticon': {
                verticalAlign: 0,
                fontSize: 12,
            },
        },
        text: {
            padding: 0,
            fontSize: theme.typography.fontSizeBase,
            fontWeight: theme.typography.typographyRegularFontWeight,
            lineHeight: theme.typography.lineHeightSm,
            '& .anticon': {
                verticalAlign: 'text-top',
            },
            whiteSpace: 'nowrap',
        },
        icon: {
            color: iconColor,
            paddingLeft: theme.spacing.xs,
            height: theme.typography.lineHeightBase,
            display: 'inline-flex',
            alignItems: 'center',
            borderTopLeftRadius: theme.borders.borderRadiusSm,
            borderBottomLeftRadius: theme.borders.borderRadiusSm,
            '& > span': {
                fontSize: 12,
            },
            '& + div': {
                borderTopLeftRadius: 0,
                borderBottomLeftRadius: 0,
                ...(closable && {
                    borderTopRightRadius: 0,
                    borderBottomRightRadius: 0,
                }),
            },
        },
        childrenWrapper: {
            paddingLeft: theme.spacing.xs,
            paddingRight: theme.spacing.xs,
            height: theme.typography.lineHeightBase,
            display: 'inline-flex',
            alignItems: 'center',
            borderRadius: theme.borders.borderRadiusSm,
            minWidth: 0,
        },
    };
}
export const Tag = forwardRef((props, forwardedRef) => {
    const { theme } = useDesignSystemTheme();
    const { color, children, closable, onClose, role = 'status', closeButtonProps, analyticsEvents, componentId, icon, onClick, ...attributes } = props;
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.tag', false);
    const isClickable = Boolean(props.onClick);
    const memoizedAnalyticsEvents = useMemo(() => analyticsEvents ??
        (emitOnView
            ? [DesignSystemEventProviderAnalyticsEventTypes.OnClick, DesignSystemEventProviderAnalyticsEventTypes.OnView]
            : [DesignSystemEventProviderAnalyticsEventTypes.OnClick]), [analyticsEvents, emitOnView]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.Tag,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
    });
    const { elementRef } = useNotifyOnFirstView({ onView: eventContext.onView });
    const mergedRef = useMergeRefs([elementRef, forwardedRef]);
    const closeButtonComponentId = componentId ? `${componentId}.close` : undefined;
    const closeButtonEventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.Button,
        componentId: closeButtonComponentId,
        analyticsEvents: [DesignSystemEventProviderAnalyticsEventTypes.OnClick],
    });
    const handleClick = useCallback((e) => {
        if (onClick) {
            eventContext.onClick(e);
            onClick(e);
        }
    }, [eventContext, onClick]);
    const handleCloseClick = useCallback((e) => {
        closeButtonEventContext.onClick(e);
        e.stopPropagation();
        if (onClose) {
            onClose();
        }
    }, [closeButtonEventContext, onClose]);
    const styles = getTagEmotionStyles(theme, color, isClickable, closable);
    return (_jsxs("div", { ref: mergedRef, role: role, onClick: handleClick, css: [styles.wrapper], ...attributes, ...addDebugOutlineIfEnabled(), ...eventContext.dataComponentProps, 
        // eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex
        tabIndex: isClickable ? 0 : -1, children: [_jsxs("div", { css: [styles.tag, styles.content, styles.text, { marginRight: 0 }], ...eventContext.dataComponentProps, children: [icon && _jsx("div", { css: [styles.icon], children: icon }), _jsx("div", { css: [styles.childrenWrapper], children: children })] }), closable && (_jsx("button", { css: styles.close, tabIndex: 0, onClick: handleCloseClick, onMouseDown: (e) => {
                    // Keeps dropdowns of any underlying select from opening.
                    e.stopPropagation();
                }, ...closeButtonProps, ...closeButtonEventContext.dataComponentProps, children: _jsx(CloseIcon, { css: { fontSize: theme.general.iconFontSize - 4 } }) }))] }));
});
//# sourceMappingURL=index.js.map