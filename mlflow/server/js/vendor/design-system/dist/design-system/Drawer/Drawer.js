import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { css, keyframes } from '@emotion/react';
import { useMergeRefs } from '@floating-ui/react';
import * as DialogPrimitive from '@radix-ui/react-dialog';
import React, { useCallback, useMemo, useRef, useState } from 'react';
import { Button } from '../Button';
import { DesignSystemEventProviderAnalyticsEventTypes, DesignSystemEventProviderComponentTypes, useDesignSystemEventComponentCallbacks, } from '../DesignSystemEventProvider';
import { ApplyDesignSystemContextOverrides } from '../DesignSystemProvider';
import { useDesignSystemContext } from '../Hooks/useDesignSystemContext';
import { useDesignSystemTheme } from '../Hooks/useDesignSystemTheme';
import { CloseIcon } from '../Icon';
import { Spacer } from '../Spacer';
import { Typography } from '../Typography';
import { getShadowScrollStyles, useDesignSystemSafexFlags, useNotifyOnFirstView } from '../utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
const DEFAULT_WIDTH = 320;
const MIN_WIDTH = 320;
const MAX_WIDTH = '90vw';
const DEFAULT_POSITION = 'right';
const ZINDEX_OVERLAY = 1;
const ZINDEX_CONTENT = ZINDEX_OVERLAY + 1;
/** Context to track if drawer is nested within a parent drawer */
const DrawerContext = React.createContext({ isParentDrawerOpen: false });
export const Content = ({ children, footer, title, width, position: positionOverride, useCustomScrollBehavior, expandContentToFullHeight, disableOpenAutoFocus, onInteractOutside, seeThrough, hideClose, closeOnClickOutside, onCloseClick, componentId = 'design_system.drawer.content', analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnView], size = 'default', ...props }) => {
    const { getPopupContainer } = useDesignSystemContext();
    const { theme } = useDesignSystemTheme();
    const horizontalContentPadding = size === 'small' ? theme.spacing.md : theme.spacing.lg;
    const [shouldContentBeFocusable, setShouldContentBeFocusable] = useState(false);
    const contentContainerRef = useRef(null);
    const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.Drawer,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
    });
    const { useNewShadows } = useDesignSystemSafexFlags();
    const { isParentDrawerOpen } = React.useContext(DrawerContext);
    const { elementRef: onViewRef } = useNotifyOnFirstView({ onView: eventContext.onView });
    const contentRef = useCallback((node) => {
        if (!node || !node.clientHeight)
            return;
        setShouldContentBeFocusable(node.scrollHeight > node.clientHeight);
    }, []);
    const mergedContentRef = useMergeRefs([contentRef, onViewRef]);
    const position = positionOverride ?? DEFAULT_POSITION;
    const overlayShow = position === 'right'
        ? keyframes({
            '0%': { transform: 'translate(100%, 0)' },
            '100%': { transform: 'translate(0, 0)' },
        })
        : keyframes({
            '0%': { transform: 'translate(-100%, 0)' },
            '100%': { transform: 'translate(0, 0)' },
        });
    const dialogPrimitiveContentStyle = css({
        color: theme.colors.textPrimary,
        backgroundColor: theme.colors.backgroundPrimary,
        boxShadow: useNewShadows
            ? theme.shadows.xl
            : 'hsl(206 22% 7% / 35%) 0px 10px 38px -10px, hsl(206 22% 7% / 20%) 0px 10px 20px -15px',
        position: 'fixed',
        top: 0,
        left: position === 'left' ? 0 : undefined,
        right: position === 'right' ? 0 : undefined,
        boxSizing: 'border-box',
        width: width ?? DEFAULT_WIDTH,
        minWidth: MIN_WIDTH,
        maxWidth: MAX_WIDTH,
        zIndex: theme.options.zIndexBase + ZINDEX_CONTENT,
        height: '100vh',
        paddingTop: size === 'small' ? theme.spacing.sm : theme.spacing.md,
        paddingLeft: 0,
        paddingBottom: 0,
        paddingRight: 0,
        overflow: 'hidden',
        '&:focus': { outline: 'none' },
        ...(isParentDrawerOpen
            ? {}
            : {
                '@media (prefers-reduced-motion: no-preference)': {
                    animation: `${overlayShow} 350ms cubic-bezier(0.16, 1, 0.3, 1)`,
                },
            }),
    });
    return (_jsxs(DialogPrimitive.Portal, { container: getPopupContainer && getPopupContainer(), children: [_jsx(DialogPrimitive.Overlay, { "data-testid": "drawer-overlay", css: {
                    backgroundColor: theme.colors.overlayOverlay,
                    position: 'fixed',
                    inset: 0,
                    // needed so that it covers the PersonaNavSidebar
                    zIndex: theme.options.zIndexBase + ZINDEX_OVERLAY,
                    opacity: seeThrough || isParentDrawerOpen ? 0 : 1,
                }, onClick: closeOnClickOutside ? onCloseClick : undefined }), _jsx(DialogPrimitive.DialogContent, { ...addDebugOutlineIfEnabled(), css: dialogPrimitiveContentStyle, style: {
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'flex-start',
                    opacity: seeThrough ? 0 : 1,
                    ...(theme.isDarkMode && {
                        borderLeft: `1px solid ${theme.colors.borderDecorative}`,
                        ...(!useNewShadows && {
                            boxShadow: 'none',
                        }),
                    }),
                }, onWheel: (e) => {
                    e.stopPropagation();
                }, onTouchMove: (e) => {
                    e.stopPropagation();
                }, "aria-hidden": seeThrough, ref: contentContainerRef, onOpenAutoFocus: (event) => {
                    if (disableOpenAutoFocus) {
                        event.preventDefault();
                    }
                }, onInteractOutside: onInteractOutside, ...props, ...eventContext.dataComponentProps, children: _jsxs(ApplyDesignSystemContextOverrides, { getPopupContainer: () => contentContainerRef.current ?? document.body, children: [(title || !hideClose) && (_jsxs("div", { css: {
                                flexGrow: 0,
                                flexShrink: 1,
                                display: 'flex',
                                flexDirection: 'row',
                                justifyContent: 'space-between',
                                alignItems: 'center',
                                paddingRight: horizontalContentPadding,
                                paddingLeft: horizontalContentPadding,
                                marginBottom: theme.spacing.sm,
                            }, children: [_jsx(DialogPrimitive.Title, { title: typeof title === 'string' ? title : undefined, asChild: typeof title === 'string', css: {
                                        flexGrow: 1,
                                        marginBottom: 0,
                                        marginTop: 0,
                                        whiteSpace: 'nowrap',
                                        overflow: 'hidden',
                                    }, children: typeof title === 'string' ? (_jsx(Typography.Title, { elementLevel: 2, level: size === 'small' ? 3 : 2, withoutMargins: true, ellipsis: true, children: title })) : (title) }), !hideClose && (_jsx(DialogPrimitive.Close, { asChild: true, css: { flexShrink: 1, marginLeft: theme.spacing.xs }, onClick: onCloseClick, children: _jsx(Button, { componentId: `${componentId}.close`, "aria-label": "Close", icon: _jsx(CloseIcon, {}), size: size === 'small' ? 'small' : undefined }) }))] })), _jsxs("div", { ref: mergedContentRef, 
                            // Needed to make drawer content focusable when scrollable for keyboard-only users to be able to focus & scroll
                            // eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex
                            tabIndex: shouldContentBeFocusable ? 0 : -1, css: {
                                // in order to have specific content in the drawer scroll with fixed title
                                // hide overflow here and remove padding on the right side; content will be responsible for setting right padding
                                // so that the scrollbar will appear in the padding right gutter
                                paddingRight: useCustomScrollBehavior ? 0 : horizontalContentPadding,
                                paddingLeft: horizontalContentPadding,
                                overflowY: useCustomScrollBehavior ? 'hidden' : 'auto',
                                height: expandContentToFullHeight ? '100%' : undefined,
                                ...(!useCustomScrollBehavior ? getShadowScrollStyles(theme) : {}),
                            }, children: [_jsx(DrawerContext.Provider, { value: { isParentDrawerOpen: true }, children: children }), !footer && _jsx(Spacer, { size: size === 'small' ? 'md' : 'lg' })] }), footer && (_jsx("div", { style: {
                                paddingTop: theme.spacing.md,
                                paddingRight: horizontalContentPadding,
                                paddingLeft: horizontalContentPadding,
                                paddingBottom: size === 'small' ? theme.spacing.md : theme.spacing.lg,
                                flexGrow: 0,
                                flexShrink: 1,
                            }, children: footer }))] }) })] }));
};
export function Root(props) {
    return _jsx(DialogPrimitive.Root, { ...props });
}
export function Trigger(props) {
    return _jsx(DialogPrimitive.Trigger, { asChild: true, ...props });
}
//# sourceMappingURL=Drawer.js.map