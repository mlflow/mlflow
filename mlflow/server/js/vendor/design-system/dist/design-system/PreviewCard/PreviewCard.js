import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { useCallback, useMemo } from 'react';
import { useDesignSystemSafexFlags, useDesignSystemTheme } from '..';
import { DesignSystemEventProviderComponentTypes, useDesignSystemEventComponentCallbacks, } from '../DesignSystemEventProvider/DesignSystemEventProvider';
import { addDebugOutlineIfEnabled } from '../utils/debug';
import { useNotifyOnFirstView } from '../utils/useNotifyOnFirstView';
export const PreviewCard = ({ icon, title, subtitle, titleActions, children, startActions, endActions, image, fullBleedImage = true, onClick, size = 'default', dangerouslyAppendEmotionCSS, componentId, analyticsEvents = [], disabled, selected, href, target, ...props }) => {
    const styles = usePreviewCardStyles({ onClick, size, disabled, fullBleedImage });
    const tabIndex = onClick && !href ? 0 : undefined;
    const role = onClick && !href ? 'button' : undefined;
    const showFooter = startActions || endActions;
    const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.PreviewCard,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
    });
    const { elementRef: previewCardRef } = useNotifyOnFirstView({
        onView: eventContext.onView,
    });
    const onClickWrapper = useCallback((e) => {
        if (onClick) {
            eventContext.onClick(e);
            onClick(e);
        }
    }, [eventContext, onClick]);
    const content = (_jsxs("div", { ...addDebugOutlineIfEnabled(), css: [styles['container'], dangerouslyAppendEmotionCSS], tabIndex: tabIndex, onClick: onClickWrapper, onKeyDown: (e) => {
            if (!onClick || disabled) {
                return;
            }
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                onClickWrapper(e);
            }
        }, role: role, "aria-disabled": disabled, "aria-pressed": selected, ...props, ref: previewCardRef, children: [image && _jsx("div", { css: styles['image'], children: image }), _jsxs("div", { css: styles['header'], children: [icon && _jsx("div", { children: icon }), _jsxs("div", { css: styles['titleWrapper'], children: [title && _jsx("div", { css: styles['title'], children: title }), subtitle && _jsx("div", { css: styles['subTitle'], children: subtitle })] }), titleActions && _jsx("div", { children: titleActions })] }), children && _jsx("div", { css: styles['childrenWrapper'], children: children }), showFooter && (_jsxs("div", { css: styles['footer'], children: [_jsx("div", { css: styles['action'], children: startActions }), _jsx("div", { css: styles['action'], children: endActions })] }))] }));
    if (href) {
        return (_jsx("a", { href: href, target: target, style: { textDecoration: 'none' }, children: content }));
    }
    return content;
};
const usePreviewCardStyles = ({ onClick, size, disabled, fullBleedImage, }) => {
    const { theme } = useDesignSystemTheme();
    const { useNewShadows } = useDesignSystemSafexFlags();
    const isInteractive = onClick !== undefined;
    const paddingSize = size === 'large' ? theme.spacing.lg : theme.spacing.md;
    return {
        container: {
            overflow: 'hidden',
            borderRadius: theme.borders.borderRadiusMd,
            border: `1px solid ${theme.colors.border}`,
            padding: paddingSize,
            color: theme.colors.textSecondary,
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
            gap: size === 'large' ? theme.spacing.md : theme.spacing.sm,
            ...(useNewShadows
                ? {
                    boxShadow: theme.shadows.sm,
                }
                : {}),
            cursor: isInteractive ? 'pointer' : 'default',
            ...(isInteractive && {
                '&[aria-disabled="true"]': {
                    pointerEvents: 'none',
                    backgroundColor: theme.colors.actionDisabledBackground,
                    borderColor: theme.colors.actionDisabledBorder,
                    color: theme.colors.actionDisabledText,
                },
                '&:hover, &:focus-within': {
                    boxShadow: useNewShadows ? theme.shadows.md : theme.general.shadowLow,
                },
                '&:active': {
                    background: theme.colors.actionTertiaryBackgroundPress,
                    borderColor: theme.colors.actionDefaultBorderHover,
                    boxShadow: useNewShadows ? theme.shadows.md : theme.general.shadowLow,
                },
                '&:focus, &[aria-pressed="true"]': {
                    outlineColor: theme.colors.actionDefaultBorderFocus,
                    outlineWidth: 2,
                    outlineOffset: -2,
                    outlineStyle: 'solid',
                    boxShadow: useNewShadows ? theme.shadows.md : theme.general.shadowLow,
                    borderColor: theme.colors.actionDefaultBorderHover,
                },
                '&:active:not(:focus):not(:focus-within)': {
                    background: 'transparent',
                    borderColor: theme.colors.border,
                },
            }),
        },
        image: {
            margin: fullBleedImage ? `-${paddingSize}px -${paddingSize}px 0` : 0,
            '& > *': {
                borderRadius: fullBleedImage ? 0 : theme.borders.borderRadiusSm,
            },
        },
        header: {
            display: 'flex',
            alignItems: 'center',
            gap: theme.spacing.sm,
        },
        title: {
            fontWeight: theme.typography.typographyBoldFontWeight,
            color: disabled ? theme.colors.actionDisabledText : theme.colors.textPrimary,
            lineHeight: theme.typography.lineHeightSm,
        },
        subTitle: {
            lineHeight: theme.typography.lineHeightSm,
        },
        titleWrapper: {
            flexGrow: 1,
            overflow: 'hidden',
        },
        childrenWrapper: {
            flexGrow: 1,
        },
        footer: {
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap',
        },
        action: {
            overflow: 'hidden',
            // to ensure focus ring is rendered
            margin: theme.spacing.md * -1,
            padding: theme.spacing.md,
        },
    };
};
//# sourceMappingURL=PreviewCard.js.map