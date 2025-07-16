import { jsx as _jsx, Fragment as _Fragment, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { useMergeRefs } from '@floating-ui/react';
import { isUndefined } from 'lodash';
import React, { useCallback, useMemo } from 'react';
import { DesignSystemEventProviderAnalyticsEventTypes, DesignSystemEventProviderComponentTypes, useDesignSystemEventComponentCallbacks, } from '../DesignSystemEventProvider/DesignSystemEventProvider';
import { getAnimationCss } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks';
import { ParagraphSkeleton, TitleSkeleton } from '../Skeleton';
import { useDesignSystemSafexFlags, useNotifyOnFirstView } from '../utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
import { safex } from '../utils/safex';
// Loading state requires a width since it'll have no content
const LOADING_STATE_DEFAULT_WIDTH = 300;
function getStyles(args) {
    const { theme, loading, width, disableHover, hasTopBar, hasBottomBar, hasHref, useNewShadows, useNewBorderRadii } = args;
    const hoverOrFocusStyle = {
        boxShadow: disableHover || loading ? '' : useNewShadows ? theme.shadows.sm : theme.general.shadowLow,
        ...(hasHref && {
            border: `1px solid ${theme.colors.actionDefaultBorderHover}`,
            ...(useNewShadows && {
                boxShadow: theme.shadows.md,
            }),
        }),
    };
    return css({
        color: theme.colors.textPrimary,
        backgroundColor: theme.colors.backgroundPrimary,
        position: 'relative',
        display: 'flex',
        justifyContent: 'flex-start',
        flexDirection: 'column',
        paddingRight: hasTopBar || hasBottomBar ? 0 : theme.spacing.md,
        paddingLeft: hasTopBar || hasBottomBar ? 0 : theme.spacing.md,
        paddingTop: hasTopBar ? 0 : theme.spacing.md,
        paddingBottom: hasBottomBar ? 0 : theme.spacing.md,
        width: width ?? 'fit-content',
        borderRadius: useNewBorderRadii ? theme.borders.borderRadiusMd : theme.legacyBorders.borderRadiusMd,
        borderColor: theme.colors.border,
        borderWidth: '1px',
        borderStyle: 'solid',
        '&:hover': hoverOrFocusStyle,
        '&:focus': hoverOrFocusStyle,
        cursor: disableHover || loading ? 'default' : 'pointer',
        ...(useNewShadows && {
            boxShadow: theme.shadows.sm,
        }),
        transition: `box-shadow 0.2s ease-in-out`,
        textDecoration: 'none !important',
        ...getAnimationCss(theme.options.enableAnimation),
    });
}
function getBottomBarStyles(theme, useNewBorderRadii) {
    return css({
        marginTop: theme.spacing.sm,
        borderBottomRightRadius: useNewBorderRadii ? theme.borders.borderRadiusSm : theme.legacyBorders.borderRadiusMd,
        borderBottomLeftRadius: useNewBorderRadii ? theme.borders.borderRadiusSm : theme.legacyBorders.borderRadiusMd,
        overflow: 'hidden',
    });
}
function getTopBarStyles(theme, useNewBorderRadii) {
    return css({
        marginBottom: theme.spacing.sm,
        borderTopRightRadius: useNewBorderRadii ? theme.borders.borderRadiusSm : theme.legacyBorders.borderRadiusMd,
        borderTopLeftRadius: useNewBorderRadii ? theme.borders.borderRadiusSm : theme.legacyBorders.borderRadiusMd,
        overflow: 'hidden',
    });
}
export const Card = ({ children, customLoadingContent, dangerouslyAppendEmotionCSS, loading, width, bottomBarContent, topBarContent, disableHover, onClick, href, navigateFn, anchorProps, componentId, analyticsEvents, shouldStartInteraction, ...dataAndAttributes }) => {
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.card', false);
    const { theme } = useDesignSystemTheme();
    const { useNewShadows, useNewBorderRadii } = useDesignSystemSafexFlags();
    const hasTopBar = !isUndefined(topBarContent);
    const hasBottomBar = !isUndefined(bottomBarContent);
    const cardStyle = css(getStyles({
        theme,
        loading,
        width,
        disableHover,
        hasBottomBar,
        hasTopBar,
        hasHref: Boolean(href),
        useNewShadows,
        useNewBorderRadii,
    }));
    const ref = React.useRef(null);
    const bottomBar = bottomBarContent ? (_jsx("div", { css: css(getBottomBarStyles(theme, useNewBorderRadii)), children: bottomBarContent })) : null;
    const topBar = topBarContent ? _jsx("div", { css: css(getTopBarStyles(theme, useNewBorderRadii)), children: topBarContent }) : null;
    const contentPadding = hasTopBar || hasBottomBar ? theme.spacing.lg : 0;
    const memoizedAnalyticsEvents = useMemo(() => analyticsEvents ??
        (emitOnView
            ? [DesignSystemEventProviderAnalyticsEventTypes.OnClick, DesignSystemEventProviderAnalyticsEventTypes.OnView]
            : [DesignSystemEventProviderAnalyticsEventTypes.OnClick]), [analyticsEvents, emitOnView]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.Card,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        shouldStartInteraction,
    });
    const { elementRef: cardRef } = useNotifyOnFirstView({ onView: eventContext.onView });
    const mergedRef = useMergeRefs([ref, cardRef]);
    const navigate = useCallback(async () => {
        if (navigateFn) {
            await navigateFn();
        }
    }, [navigateFn]);
    const handleClick = useCallback(async (e) => {
        eventContext.onClick(e);
        await navigate();
        onClick?.(e);
        ref.current?.blur();
    }, [navigate, eventContext, onClick]);
    const handleSelection = useCallback(async (e) => {
        eventContext.onClick(e);
        e.preventDefault();
        await navigate();
        onClick?.(e);
    }, [navigate, eventContext, onClick]);
    const content = (_jsx("div", { ref: mergedRef, 
        // eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex
        tabIndex: 0, ...addDebugOutlineIfEnabled(), css: href ? [] : [cardStyle, dangerouslyAppendEmotionCSS], onClick: loading || href ? undefined : handleClick, ...dataAndAttributes, ...(href && {
            onKeyDown: async (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    await handleSelection(e);
                }
                dataAndAttributes.onKeyDown?.(e);
            },
        }), ...eventContext.dataComponentProps, children: loading ? (_jsx(DefaultCardLoadingContent, { width: width, customLoadingContent: customLoadingContent })) : (_jsxs(_Fragment, { children: [topBar, _jsx("div", { css: { padding: `0px ${contentPadding}px`, flexGrow: 1 }, children: children }), bottomBar] })) }));
    return href ? (_jsx("a", { css: [cardStyle, dangerouslyAppendEmotionCSS], href: href, ...anchorProps, children: content })) : (content);
};
function DefaultCardLoadingContent({ customLoadingContent, width }) {
    if (customLoadingContent) {
        return _jsx(_Fragment, { children: customLoadingContent });
    }
    return (_jsxs("div", { css: { width: width ?? LOADING_STATE_DEFAULT_WIDTH }, children: [_jsx(TitleSkeleton, { label: "Loading...", style: { width: '50%' } }), [...Array(3).keys()].map((i) => (_jsx(ParagraphSkeleton, { label: "Loading..." }, i)))] }));
}
//# sourceMappingURL=Card.js.map