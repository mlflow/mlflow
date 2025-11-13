import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "@emotion/react/jsx-runtime";
import { Global, css, keyframes } from '@emotion/react';
import React, { createContext, forwardRef, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { ResizableBox } from 'react-resizable';
import { Button } from '../Button';
import { useDesignSystemTheme } from '../Hooks/useDesignSystemTheme';
import { ChevronLeftIcon, ChevronRightIcon, CloseIcon } from '../Icon';
import { Typography } from '../Typography';
import { useDesignSystemSafexFlags } from '../utils';
import { importantify } from '../utils/css-utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
import { useMediaQuery } from '../utils/useMediaQuery';
/**
 * `ResizableBox` passes `handleAxis` to the element used as handle. We need to wrap the handle to prevent
 * `handleAxis becoming an attribute on the div element.
 */
const ResizablePanelHandle = forwardRef(function ResizablePanelHandle({ handleAxis, children, ...otherProps }, ref) {
    return (_jsx("div", { ref: ref, ...otherProps, children: children }));
});
const DEFAULT_WIDTH = 200;
const ContentContextDefaults = {
    openPanelId: undefined,
    closable: true,
    destroyInactivePanels: false,
    setIsClosed: () => { },
};
const SidebarContextDefaults = {
    position: 'left',
};
const ContentContext = createContext(ContentContextDefaults);
const SidebarContext = createContext(SidebarContextDefaults);
export function Nav({ children, dangerouslyAppendEmotionCSS }) {
    const { theme } = useDesignSystemTheme();
    return (_jsx("nav", { css: [
            {
                display: 'flex',
                flexDirection: 'column',
                gap: theme.spacing.xs,
                padding: theme.spacing.xs,
            },
            dangerouslyAppendEmotionCSS,
        ], children: children }));
}
export const NavButton = React.forwardRef(({ active, disabled, icon, onClick, children, dangerouslyAppendEmotionCSS, 'aria-label': ariaLabel, ...restProps }, ref) => {
    const { theme } = useDesignSystemTheme();
    return (_jsx("div", { css: [
            active
                ? importantify({
                    borderRadius: theme.borders.borderRadiusSm,
                    background: theme.colors.actionDefaultBackgroundPress,
                    button: {
                        '&:enabled:not(:hover):not(:active) > .anticon': { color: theme.colors.actionTertiaryTextPress },
                    },
                })
                : undefined,
            dangerouslyAppendEmotionCSS,
        ], children: _jsx(Button, { ref: ref, icon: icon, onClick: onClick, disabled: disabled, "aria-label": ariaLabel, ...restProps, children: children }) }));
});
const TOGGLE_BUTTON_Z_INDEX = 100;
const COMPACT_CONTENT_Z_INDEX = 50;
const ToggleButton = ({ isExpanded, position, toggleIsExpanded, componentId }) => {
    const { theme } = useDesignSystemTheme();
    const positionStyle = useMemo(() => {
        if (position === 'right') {
            return isExpanded
                ? { right: DEFAULT_WIDTH, transform: 'translateX(+50%)' }
                : { left: 0, transform: 'translateX(-50%)' };
        }
        else {
            return isExpanded
                ? { left: DEFAULT_WIDTH, transform: 'translateX(-50%)' }
                : { right: 0, transform: 'translateX(+50%)' };
        }
    }, [isExpanded, position]);
    const ToggleIcon = useMemo(() => {
        if (position === 'right') {
            return isExpanded ? ChevronRightIcon : ChevronLeftIcon;
        }
        else {
            return isExpanded ? ChevronLeftIcon : ChevronRightIcon;
        }
    }, [isExpanded, position]);
    return (_jsxs("div", { css: {
            position: 'absolute',
            top: 0,
            height: 46,
            display: 'flex',
            alignItems: 'center',
            zIndex: TOGGLE_BUTTON_Z_INDEX,
            ...positionStyle,
        }, children: [_jsx("div", { css: {
                    borderRadius: '100%',
                    width: theme.spacing.lg,
                    height: theme.spacing.lg,
                    backgroundColor: theme.colors.backgroundPrimary,
                    position: 'absolute',
                } }), _jsx(Button, { componentId: componentId, css: {
                    borderRadius: '100%',
                    '&&': {
                        padding: '0px !important',
                        width: `${theme.spacing.lg}px !important`,
                    },
                }, onClick: toggleIsExpanded, size: "small", "aria-label": isExpanded ? 'hide sidebar' : 'expand sidebar', "aria-expanded": isExpanded, children: _jsx(ToggleIcon, {}) })] }));
};
const getContentAnimation = (width) => {
    const showAnimation = keyframes `
  from { opacity: 0 }
  80%  { opacity: 0 }
  to   { opacity: 1 }`;
    const openAnimation = keyframes `
  from { width: 50px }
  to   { width: ${width}px }`;
    return {
        open: `${openAnimation} .2s cubic-bezier(0, 0, 0.2, 1)`,
        show: `${showAnimation} .25s linear`,
    };
};
export function Content({ disableResize, openPanelId, closable = true, onClose, onResizeStart, onResizeStop, width, minWidth, maxWidth, destroyInactivePanels = false, children, dangerouslyAppendEmotionCSS, enableCompact, resizeBoxStyle, noSideBorder, hideResizeHandle, componentId, }) {
    const { theme } = useDesignSystemTheme();
    const isCompact = useMediaQuery({ query: `not (min-width: ${theme.responsive.breakpoints.sm}px)` }) && enableCompact;
    const defaultAnimation = useMemo(() => getContentAnimation(isCompact ? DEFAULT_WIDTH : width || DEFAULT_WIDTH), [isCompact, width]);
    // specifically for non closable panel in compact mode
    const [isExpanded, setIsExpanded] = useState(true);
    // hide the panel in compact mode when the panel is not closable and collapsed
    const isNotExpandedStyle = css(isCompact && !closable && !isExpanded && { display: 'none' });
    const sidebarContext = useContext(SidebarContext);
    const onCloseRef = useRef(onClose);
    const resizeHandleStyle = sidebarContext.position === 'right' ? { left: 0 } : { right: 0 };
    const [dragging, setDragging] = useState(false);
    const isPanelClosed = openPanelId == null;
    const [animation, setAnimation] = useState(isPanelClosed ? defaultAnimation : undefined);
    const compactStyle = css(isCompact && {
        position: 'absolute',
        zIndex: COMPACT_CONTENT_Z_INDEX,
        left: sidebarContext.position === 'left' && closable ? '100%' : undefined,
        right: sidebarContext.position === 'right' && closable ? '100%' : undefined,
        borderRight: sidebarContext.position === 'left' && !noSideBorder ? `1px solid ${theme.colors.border}` : undefined,
        borderLeft: sidebarContext.position === 'right' && !noSideBorder ? `1px solid ${theme.colors.border}` : undefined,
        backgroundColor: theme.colors.backgroundPrimary,
        width: DEFAULT_WIDTH,
        // shift to the top due to border
        top: -1,
    });
    const hiddenPanelStyle = css(isPanelClosed && { display: 'none' });
    const containerStyle = css({
        animation: animation?.open,
        direction: sidebarContext.position === 'right' ? 'rtl' : 'ltr',
        position: 'relative',
        borderWidth: sidebarContext.position === 'right'
            ? `0 ${noSideBorder ? 0 : theme.general.borderWidth}px 0 0 `
            : `0 0 0 ${noSideBorder ? 0 : theme.general.borderWidth}px`,
        borderStyle: 'inherit',
        borderColor: 'inherit',
        boxSizing: 'content-box',
    });
    const highlightedBorderStyle = sidebarContext.position === 'right'
        ? css({ borderLeft: `2px solid ${theme.colors.actionDefaultBorderHover}` })
        : css({ borderRight: `2px solid ${theme.colors.actionDefaultBorderHover}` });
    useEffect(() => {
        onCloseRef.current = onClose;
    }, [onClose]);
    // For non closable panel, reset expanded state to true so that the panel stays open
    // the next time the screen goes into compact mode.
    useEffect(() => {
        if (!closable && enableCompact && !isCompact) {
            setIsExpanded(true);
        }
    }, [isCompact, closable, defaultAnimation, enableCompact]);
    const value = useMemo(() => ({
        openPanelId,
        closable,
        destroyInactivePanels,
        setIsClosed: () => {
            onCloseRef.current?.();
            if (!animation) {
                setAnimation(defaultAnimation);
            }
        },
    }), [openPanelId, closable, defaultAnimation, animation, destroyInactivePanels]);
    return (_jsx(ContentContext.Provider, { value: value, children: disableResize || isCompact ? (_jsxs(_Fragment, { children: [_jsx("div", { css: [
                        css({ width: width || '100%', height: '100%', overflow: 'hidden' }, containerStyle, compactStyle),
                        dangerouslyAppendEmotionCSS,
                        hiddenPanelStyle,
                        isNotExpandedStyle,
                    ], "aria-hidden": isPanelClosed, children: _jsx("div", { css: { opacity: 1, height: '100%', animation: animation?.show, direction: 'ltr' }, children: children }) }), !closable && isCompact && (_jsx("div", { css: {
                        width: !isExpanded ? theme.spacing.md : undefined,
                        marginRight: isExpanded ? theme.spacing.md : undefined,
                        position: 'relative',
                    }, children: _jsx(ToggleButton, { componentId: componentId ? `${componentId}.toggle` : 'sidebar-toggle', isExpanded: isExpanded, position: sidebarContext.position || 'left', toggleIsExpanded: () => setIsExpanded((prev) => !prev) }) }))] })) : (_jsxs(_Fragment, { children: [dragging && (_jsx(Global, { styles: {
                        'body, :host': {
                            userSelect: 'none',
                        },
                    } })), _jsx(ResizableBox, { style: resizeBoxStyle, width: width || DEFAULT_WIDTH, height: undefined, axis: "x", resizeHandles: sidebarContext.position === 'right' ? ['w'] : ['e'], minConstraints: [minWidth ?? DEFAULT_WIDTH, 150], maxConstraints: [maxWidth ?? 800, 150], onResizeStart: (_, { size }) => {
                        onResizeStart?.(size.width);
                        setDragging(true);
                    }, onResizeStop: (_, { size }) => {
                        onResizeStop?.(size.width);
                        setDragging(false);
                    }, handle: hideResizeHandle ? (
                    // Passing null shows default handle from react-resizable
                    _jsx(_Fragment, {})) : (_jsx(ResizablePanelHandle, { css: css({
                            width: 10,
                            height: '100%',
                            position: 'absolute',
                            top: 0,
                            cursor: sidebarContext.position === 'right' ? 'w-resize' : 'e-resize',
                            '&:hover': highlightedBorderStyle,
                            ...resizeHandleStyle,
                        }, dragging && highlightedBorderStyle) })), css: [containerStyle, hiddenPanelStyle], "aria-hidden": isPanelClosed, children: _jsx("div", { css: [
                            {
                                opacity: 1,
                                animation: animation?.show,
                                direction: 'ltr',
                                height: '100%',
                            },
                            dangerouslyAppendEmotionCSS,
                        ], children: children }) })] })) }));
}
export function Panel({ panelId, children, forceRender = false, dangerouslyAppendEmotionCSS, ...delegated }) {
    const { openPanelId, destroyInactivePanels } = useContext(ContentContext);
    const hasOpenedPanelRef = useRef(false);
    const isPanelOpen = openPanelId === panelId;
    if (isPanelOpen && !hasOpenedPanelRef.current) {
        hasOpenedPanelRef.current = true;
    }
    if ((destroyInactivePanels || !hasOpenedPanelRef.current) && !isPanelOpen && !forceRender)
        return null;
    return (_jsx("div", { css: [
            { display: 'flex', height: '100%', flexDirection: 'column' },
            dangerouslyAppendEmotionCSS,
            !isPanelOpen && { display: 'none' },
        ], "aria-hidden": !isPanelOpen, ...delegated, children: children }));
}
export function PanelHeader({ children, dangerouslyAppendEmotionCSS, componentId }) {
    const { theme } = useDesignSystemTheme();
    const contentContext = useContext(ContentContext);
    return (_jsxs("div", { css: [
            {
                display: 'flex',
                paddingLeft: 8,
                paddingRight: 4,
                alignItems: 'center',
                minHeight: theme.general.heightSm,
                justifyContent: 'space-between',
                fontWeight: theme.typography.typographyBoldFontWeight,
                color: theme.colors.textPrimary,
            },
            dangerouslyAppendEmotionCSS,
        ], children: [_jsx("div", { css: { width: contentContext.closable ? `calc(100% - ${theme.spacing.lg}px)` : '100%' }, children: _jsx(Typography.Title, { level: 4, css: {
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        '&&': {
                            margin: 0,
                        },
                    }, children: children }) }), contentContext.closable ? (_jsx("div", { children: _jsx(Button, { componentId: componentId ? `${componentId}.close` : 'codegen_design-system_src_design-system_sidebar_sidebar.tsx_427', size: "small", icon: _jsx(CloseIcon, {}), "aria-label": "Close", onClick: () => {
                        contentContext.setIsClosed();
                    } }) })) : null] }));
}
export function PanelHeaderTitle({ title, dangerouslyAppendEmotionCSS }) {
    return (_jsx("div", { title: title, css: [
            {
                alignSelf: 'center',
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
            },
            dangerouslyAppendEmotionCSS,
        ], children: title }));
}
export function PanelHeaderButtons({ children, dangerouslyAppendEmotionCSS }) {
    const { theme } = useDesignSystemTheme();
    return (_jsx("div", { css: [
            { display: 'flex', alignItems: 'center', gap: theme.spacing.xs, paddingRight: theme.spacing.xs },
            dangerouslyAppendEmotionCSS,
        ], children: children }));
}
export function PanelBody({ children, dangerouslyAppendEmotionCSS }) {
    const { theme } = useDesignSystemTheme();
    const [shouldBeFocusable, setShouldBeFocusable] = useState(false);
    const bodyRef = useRef(null);
    useEffect(() => {
        const ref = bodyRef.current;
        if (ref) {
            if (ref.scrollHeight > ref.clientHeight) {
                setShouldBeFocusable(true);
            }
            else {
                setShouldBeFocusable(false);
            }
        }
    }, []);
    return (_jsx("div", { ref: bodyRef, 
        // Needed to make panel body content focusable when scrollable for keyboard-only users to be able to focus & scroll
        // eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex
        tabIndex: shouldBeFocusable ? 0 : -1, css: [
            {
                height: '100%',
                overflowX: 'hidden',
                overflowY: 'auto',
                padding: '0 8px',
                colorScheme: theme.isDarkMode ? 'dark' : 'light',
            },
            dangerouslyAppendEmotionCSS,
        ], children: children }));
}
export const Sidebar = /* #__PURE__ */ (() => {
    function Sidebar({ position, children, dangerouslyAppendEmotionCSS, ...dataProps }) {
        const { theme } = useDesignSystemTheme();
        const { useNewBorderColors } = useDesignSystemSafexFlags();
        const value = useMemo(() => {
            return {
                position: position || 'left',
            };
        }, [position]);
        return (_jsx(SidebarContext.Provider, { value: value, children: _jsx("div", { ...addDebugOutlineIfEnabled(), ...dataProps, css: [
                    {
                        display: 'flex',
                        height: '100%',
                        backgroundColor: theme.colors.backgroundPrimary,
                        flexDirection: position === 'right' ? 'row-reverse' : 'row',
                        borderStyle: 'solid',
                        borderColor: useNewBorderColors ? theme.colors.border : theme.colors.borderDecorative,
                        borderWidth: position === 'right'
                            ? `0 0 0 ${theme.general.borderWidth}px`
                            : `0px ${theme.general.borderWidth}px 0 0`,
                        boxSizing: 'content-box',
                        position: 'relative',
                    },
                    dangerouslyAppendEmotionCSS,
                ], children: children }) }));
    }
    Sidebar.Content = Content;
    Sidebar.Nav = Nav;
    Sidebar.NavButton = NavButton;
    Sidebar.Panel = Panel;
    Sidebar.PanelHeader = PanelHeader;
    Sidebar.PanelHeaderTitle = PanelHeaderTitle;
    Sidebar.PanelHeaderButtons = PanelHeaderButtons;
    Sidebar.PanelBody = PanelBody;
    return Sidebar;
})();
//# sourceMappingURL=Sidebar.js.map