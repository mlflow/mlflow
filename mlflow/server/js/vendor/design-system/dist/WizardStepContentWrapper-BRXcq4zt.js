import { jsx, jsxs, Fragment } from '@emotion/react/jsx-runtime';
import isUndefined from 'lodash/isUndefined';
import React__default, { useLayoutEffect, useEffect, useState, forwardRef, useMemo, useContext, useRef, createContext, useCallback } from 'react';
import { u as useDesignSystemTheme, i as importantify, B as Button, T as Typography, C as CloseIcon, q as ChevronRightIcon, o as ChevronLeftIcon, a as addDebugOutlineIfEnabled, W as WarningIcon, k as DangerIcon, w as LoadingIcon, x as CheckIcon, y as DEFAULT_SPACING_UNIT, z as getShadowScrollStyles, A as Tooltip, I as InfoSmallIcon, E as Modal, R as Root$1, l as Trigger$1, G as ListIcon, m as Content$2 } from './Tabs-BE1SgcRI.js';
import { css, Global, keyframes, createElement } from '@emotion/react';
import 'antd';
import '@radix-ui/react-tooltip-patch';
import '@ant-design/icons';
import 'classnames';
import '@radix-ui/react-dialog';
import '@floating-ui/react';
import 'lodash/uniqueId';
import 'lodash/isNil';
import '@radix-ui/react-popover';
import '@radix-ui/react-context-menu';
import '@radix-ui/react-dropdown-menu';
import 'tabbable';
import 'react-hook-form';
import '@floating-ui/dom';
import 'react-dom';
import '@radix-ui/react-hover-card';
import '@radix-ui/react-navigation-menu';
import '@radix-ui/react-toast';
import { ResizableBox } from 'react-resizable';
import '@radix-ui/react-slider';
import '@radix-ui/react-toggle';
import 'chroma-js';
import noop from 'lodash/noop';
import pick from 'lodash/pick';
import compact from 'lodash/compact';
import 'lodash/isEqual';

const Spacer = ({ size = 'md', shrinks, ...props })=>{
    const { theme } = useDesignSystemTheme();
    const spacingValues = {
        xs: theme.spacing.xs,
        sm: theme.spacing.sm,
        md: theme.spacing.md,
        lg: theme.spacing.lg
    };
    return /*#__PURE__*/ jsx("div", {
        css: /*#__PURE__*/ css({
            height: spacingValues[size],
            ...shrinks === false ? {
                flexShrink: 0
            } : undefined
        }),
        ...props
    });
};

/**
 * Copied from usehooks-ts.
 * Custom hook for using either `useLayoutEffect` or `useEffect` based on the environment (client-side or server-side).
 *
 * [Documentation](https://usehooks-ts.com/react-hook/use-isomorphic-layout-effect)
 *
 * Example:
 * ```
 * useIsomorphicLayoutEffect(() => {
 * // Code to be executed during the layout phase on the client side
 * }, [dependency1, dependency2]);
 * ```
 */ const useIsomorphicLayoutEffect = typeof window !== 'undefined' ? useLayoutEffect : useEffect;

const IS_SERVER = typeof window === 'undefined';
/**
 * Copied from usehooks-ts.
 * Custom hook for tracking the state of a media query. Returns The current state of the media query (true if the query matches, false otherwise).
 *
 * [Documentation](https://usehooks-ts.com/react-hook/use-media-query)
 *
 * [MDN Match Media](https://developer.mozilla.org/en-US/docs/Web/API/Window/matchMedia)
 *
 * Example:
 *
 * `const isSmallScreen = useMediaQuery('(max-width: 600px)');`
 *
 * Use `isSmallScreen` to conditionally apply styles or logic based on the screen size.
 */ function useMediaQuery({ query, options }) {
    // TODO: Refactor this code after the deprecated signature has been removed.
    const defaultValue = typeof options === 'boolean' ? options : options?.defaultValue ?? false;
    const initializeWithValue = typeof options === 'boolean' ? undefined : options?.initializeWithValue ?? undefined;
    const [matches, setMatches] = useState(()=>{
        if (initializeWithValue) {
            return getMatches(query);
        }
        return defaultValue;
    });
    const getMatches = (query)=>{
        if (IS_SERVER) {
            return defaultValue;
        }
        return window.matchMedia(query).matches;
    };
    /** Handles the change event of the media query. */ function handleChange() {
        setMatches(getMatches(query));
    }
    useIsomorphicLayoutEffect(()=>{
        const matchMedia = window.matchMedia(query);
        // Triggered at the first client-side load and if query changes
        handleChange();
        // Use deprecated `addListener` and `removeListener` to support Safari < 14 (#135)
        if (matchMedia.addListener) {
            matchMedia.addListener(handleChange);
        } else {
            matchMedia.addEventListener('change', handleChange);
        }
        return ()=>{
            if (matchMedia.removeListener) {
                matchMedia.removeListener(handleChange);
            } else {
                matchMedia.removeEventListener('change', handleChange);
            }
        };
    }, [
        query
    ]);
    return matches;
}

/**
 * `ResizableBox` passes `handleAxis` to the element used as handle. We need to wrap the handle to prevent
 * `handleAxis becoming an attribute on the div element.
 */ const ResizablePanelHandle = /*#__PURE__*/ forwardRef(function ResizablePanelHandle({ handleAxis, children, ...otherProps }, ref) {
    return /*#__PURE__*/ jsx("div", {
        ref: ref,
        ...otherProps,
        children: children
    });
});
const DEFAULT_WIDTH = 200;
const ContentContextDefaults = {
    openPanelId: undefined,
    closable: true,
    destroyInactivePanels: false,
    setIsClosed: ()=>{}
};
const SidebarContextDefaults = {
    position: 'left'
};
const ContentContext = /*#__PURE__*/ createContext(ContentContextDefaults);
const SidebarContext = /*#__PURE__*/ createContext(SidebarContextDefaults);
function Nav({ children, dangerouslyAppendEmotionCSS }) {
    const { theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsx("nav", {
        css: [
            {
                display: 'flex',
                flexDirection: 'column',
                gap: theme.spacing.xs,
                padding: theme.spacing.xs
            },
            dangerouslyAppendEmotionCSS
        ],
        children: children
    });
}
const NavButton = /*#__PURE__*/ React__default.forwardRef(// eslint-disable-next-line react-component-name/react-component-name -- TODO(FEINF-4716)
({ active, disabled, icon, onClick, children, dangerouslyAppendEmotionCSS, 'aria-label': ariaLabel, ...restProps }, ref)=>{
    const { theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsx("div", {
        css: [
            active ? importantify({
                borderRadius: theme.borders.borderRadiusSm,
                background: theme.colors.actionDefaultBackgroundPress,
                button: {
                    '&:enabled:not(:hover):not(:active) > .anticon': {
                        color: theme.colors.actionTertiaryTextPress
                    }
                }
            }) : undefined,
            dangerouslyAppendEmotionCSS
        ],
        children: /*#__PURE__*/ jsx(Button, {
            ref: ref,
            icon: icon,
            onClick: onClick,
            disabled: disabled,
            "aria-label": ariaLabel,
            ...restProps,
            children: children
        })
    });
});
const TOGGLE_BUTTON_Z_INDEX = 100;
const COMPACT_CONTENT_Z_INDEX = 50;
const ToggleButton = ({ isExpanded, position, toggleIsExpanded, componentId })=>{
    const { theme } = useDesignSystemTheme();
    const positionStyle = useMemo(()=>{
        if (position === 'right') {
            return isExpanded ? {
                right: DEFAULT_WIDTH,
                transform: 'translateX(+50%)'
            } : {
                left: 0,
                transform: 'translateX(-50%)'
            };
        } else {
            return isExpanded ? {
                left: DEFAULT_WIDTH,
                transform: 'translateX(-50%)'
            } : {
                right: 0,
                transform: 'translateX(+50%)'
            };
        }
    }, [
        isExpanded,
        position
    ]);
    const ToggleIcon = useMemo(()=>{
        if (position === 'right') {
            return isExpanded ? ChevronRightIcon : ChevronLeftIcon;
        } else {
            return isExpanded ? ChevronLeftIcon : ChevronRightIcon;
        }
    }, [
        isExpanded,
        position
    ]);
    return /*#__PURE__*/ jsxs("div", {
        css: {
            position: 'absolute',
            top: 0,
            height: 46,
            display: 'flex',
            alignItems: 'center',
            zIndex: TOGGLE_BUTTON_Z_INDEX,
            ...positionStyle
        },
        children: [
            /*#__PURE__*/ jsx("div", {
                css: {
                    borderRadius: '100%',
                    width: theme.spacing.lg,
                    height: theme.spacing.lg,
                    backgroundColor: theme.colors.backgroundPrimary,
                    position: 'absolute'
                }
            }),
            /*#__PURE__*/ jsx(Button, {
                componentId: componentId,
                css: {
                    borderRadius: '100%',
                    '&&': {
                        padding: '0px !important',
                        width: `${theme.spacing.lg}px !important`
                    }
                },
                onClick: toggleIsExpanded,
                size: "small",
                "aria-label": isExpanded ? 'hide sidebar' : 'expand sidebar',
                "aria-expanded": isExpanded,
                children: /*#__PURE__*/ jsx(ToggleIcon, {})
            })
        ]
    });
};
const getContentAnimation = (width)=>{
    const showAnimation = /*#__PURE__*/ keyframes("from{opacity:0}80%{opacity:0}to{opacity:1}");
    const openAnimation = /*#__PURE__*/ keyframes("from{width:50px}to{width:", width, "px}");
    return {
        open: `${openAnimation} .2s cubic-bezier(0, 0, 0.2, 1)`,
        show: `${showAnimation} .25s linear`
    };
};
const Content$1 = /*#__PURE__*/ forwardRef(function Content({ disableResize, openPanelId, closable = true, onClose, onResizeStart, onResizeStop, width, minWidth, maxWidth, destroyInactivePanels = false, children, dangerouslyAppendEmotionCSS, enableCompact, resizeBoxStyle, noSideBorder, hideResizeHandle, componentId }, ref) {
    const { theme } = useDesignSystemTheme();
    const isCompact = useMediaQuery({
        query: `not (min-width: ${theme.responsive.breakpoints.sm}px)`
    }) && enableCompact;
    const defaultAnimation = useMemo(()=>getContentAnimation(isCompact ? DEFAULT_WIDTH : width || DEFAULT_WIDTH), [
        isCompact,
        width
    ]);
    // specifically for non closable panel in compact mode
    const [isExpanded, setIsExpanded] = useState(true);
    // hide the panel in compact mode when the panel is not closable and collapsed
    const isNotExpandedStyle = /*#__PURE__*/ css(isCompact && !closable && !isExpanded && {
        display: 'none'
    });
    const sidebarContext = useContext(SidebarContext);
    const onCloseRef = useRef(onClose);
    const resizeHandleStyle = sidebarContext.position === 'right' ? {
        left: 0
    } : {
        right: 0
    };
    const [dragging, setDragging] = useState(false);
    const isPanelClosed = openPanelId == null;
    const [animation, setAnimation] = useState(isPanelClosed ? defaultAnimation : undefined);
    const compactStyle = /*#__PURE__*/ css(isCompact && {
        position: 'absolute',
        zIndex: COMPACT_CONTENT_Z_INDEX,
        left: sidebarContext.position === 'left' && closable ? '100%' : undefined,
        right: sidebarContext.position === 'right' && closable ? '100%' : undefined,
        borderRight: sidebarContext.position === 'left' && !noSideBorder ? `1px solid ${theme.colors.border}` : undefined,
        borderLeft: sidebarContext.position === 'right' && !noSideBorder ? `1px solid ${theme.colors.border}` : undefined,
        backgroundColor: theme.colors.backgroundPrimary,
        width: DEFAULT_WIDTH,
        // shift to the top due to border
        top: -1
    });
    const hiddenPanelStyle = /*#__PURE__*/ css(isPanelClosed && {
        display: 'none'
    });
    const containerStyle = /*#__PURE__*/ css({
        animation: animation?.open,
        direction: sidebarContext.position === 'right' ? 'rtl' : 'ltr',
        position: 'relative',
        borderWidth: sidebarContext.position === 'right' ? `0 ${noSideBorder ? 0 : theme.general.borderWidth}px 0 0 ` : `0 0 0 ${noSideBorder ? 0 : theme.general.borderWidth}px`,
        borderStyle: 'inherit',
        borderColor: 'inherit',
        boxSizing: 'content-box'
    });
    const highlightedBorderStyle = sidebarContext.position === 'right' ? /*#__PURE__*/ css({
        borderLeft: `2px solid ${theme.colors.actionDefaultBorderHover}`
    }) : /*#__PURE__*/ css({
        borderRight: `2px solid ${theme.colors.actionDefaultBorderHover}`
    });
    useEffect(()=>{
        onCloseRef.current = onClose;
    }, [
        onClose
    ]);
    // For non closable panel, reset expanded state to true so that the panel stays open
    // the next time the screen goes into compact mode.
    useEffect(()=>{
        if (!closable && enableCompact && !isCompact) {
            setIsExpanded(true);
        }
    }, [
        isCompact,
        closable,
        defaultAnimation,
        enableCompact
    ]);
    const value = useMemo(()=>({
            openPanelId,
            closable,
            destroyInactivePanels,
            setIsClosed: ()=>{
                onCloseRef.current?.();
                if (!animation) {
                    setAnimation(defaultAnimation);
                }
            }
        }), [
        openPanelId,
        closable,
        defaultAnimation,
        animation,
        destroyInactivePanels
    ]);
    return /*#__PURE__*/ jsx(ContentContext.Provider, {
        value: value,
        children: disableResize || isCompact ? /*#__PURE__*/ jsxs(Fragment, {
            children: [
                /*#__PURE__*/ jsx("div", {
                    css: [
                        /*#__PURE__*/ css({
                            width: width || '100%',
                            height: '100%',
                            overflow: 'hidden'
                        }, containerStyle, compactStyle),
                        dangerouslyAppendEmotionCSS,
                        hiddenPanelStyle,
                        isNotExpandedStyle
                    ],
                    "aria-hidden": isPanelClosed,
                    children: /*#__PURE__*/ jsx("div", {
                        css: {
                            opacity: 1,
                            height: '100%',
                            animation: animation?.show,
                            direction: 'ltr'
                        },
                        children: children
                    })
                }),
                !closable && isCompact && /*#__PURE__*/ jsx("div", {
                    css: {
                        width: !isExpanded ? theme.spacing.md : undefined,
                        marginRight: isExpanded ? theme.spacing.md : undefined,
                        position: 'relative'
                    },
                    children: /*#__PURE__*/ jsx(ToggleButton, {
                        componentId: componentId ? `${componentId}.toggle` : 'sidebar-toggle',
                        isExpanded: isExpanded,
                        position: sidebarContext.position || 'left',
                        toggleIsExpanded: ()=>setIsExpanded((prev)=>!prev)
                    })
                })
            ]
        }) : /*#__PURE__*/ jsxs(Fragment, {
            children: [
                dragging && /*#__PURE__*/ jsx(Global, {
                    styles: [
                        {
                            'body, :host': {
                                userSelect: 'none'
                            }
                        }
                    ]
                }),
                /*#__PURE__*/ jsx(ResizableBox, {
                    style: resizeBoxStyle,
                    width: width || DEFAULT_WIDTH,
                    height: undefined,
                    axis: "x",
                    resizeHandles: sidebarContext.position === 'right' ? [
                        'w'
                    ] : [
                        'e'
                    ],
                    minConstraints: [
                        minWidth ?? DEFAULT_WIDTH,
                        150
                    ],
                    maxConstraints: [
                        maxWidth ?? 800,
                        150
                    ],
                    onResizeStart: (_, { size })=>{
                        onResizeStart?.(size.width);
                        setDragging(true);
                    },
                    onResizeStop: (_, { size })=>{
                        onResizeStop?.(size.width);
                        setDragging(false);
                    },
                    handle: hideResizeHandle ? // Passing null shows default handle from react-resizable
                    /*#__PURE__*/ jsx(Fragment, {}) : /*#__PURE__*/ jsx(ResizablePanelHandle, {
                        css: /*#__PURE__*/ css({
                            width: 4,
                            height: '100%',
                            position: 'absolute',
                            top: 0,
                            cursor: sidebarContext.position === 'right' ? 'w-resize' : 'e-resize',
                            '&:hover': highlightedBorderStyle,
                            ...resizeHandleStyle
                        }, dragging && highlightedBorderStyle)
                    }),
                    css: [
                        containerStyle,
                        hiddenPanelStyle
                    ],
                    "aria-hidden": isPanelClosed,
                    children: /*#__PURE__*/ jsx("div", {
                        ref: ref,
                        css: [
                            {
                                opacity: 1,
                                animation: animation?.show,
                                direction: 'ltr',
                                height: '100%'
                            },
                            dangerouslyAppendEmotionCSS
                        ],
                        children: children
                    })
                })
            ]
        })
    });
});
function Panel({ panelId, children, forceRender = false, dangerouslyAppendEmotionCSS, ...delegated }) {
    const { openPanelId, destroyInactivePanels } = useContext(ContentContext);
    const hasOpenedPanelRef = useRef(false);
    const isPanelOpen = openPanelId === panelId;
    if (isPanelOpen && !hasOpenedPanelRef.current) {
        hasOpenedPanelRef.current = true;
    }
    if ((destroyInactivePanels || !hasOpenedPanelRef.current) && !isPanelOpen && !forceRender) return null;
    return /*#__PURE__*/ jsx("div", {
        css: [
            {
                display: 'flex',
                height: '100%',
                flexDirection: 'column'
            },
            dangerouslyAppendEmotionCSS,
            !isPanelOpen && {
                display: 'none'
            }
        ],
        "aria-hidden": !isPanelOpen,
        ...delegated,
        children: children
    });
}
function PanelHeader({ children, dangerouslyAppendEmotionCSS, componentId }) {
    const { theme } = useDesignSystemTheme();
    const contentContext = useContext(ContentContext);
    return /*#__PURE__*/ jsxs("div", {
        css: [
            {
                display: 'flex',
                paddingLeft: 8,
                paddingRight: 4,
                alignItems: 'center',
                minHeight: theme.general.heightSm,
                justifyContent: 'space-between',
                fontWeight: theme.typography.typographyBoldFontWeight,
                color: theme.colors.textPrimary
            },
            dangerouslyAppendEmotionCSS
        ],
        children: [
            /*#__PURE__*/ jsx("div", {
                css: {
                    width: contentContext.closable ? `calc(100% - ${theme.spacing.lg}px)` : '100%'
                },
                children: /*#__PURE__*/ jsx(Typography.Title, {
                    level: 4,
                    css: {
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        '&&': {
                            margin: 0
                        }
                    },
                    children: children
                })
            }),
            contentContext.closable ? /*#__PURE__*/ jsx("div", {
                children: /*#__PURE__*/ jsx(Button, {
                    componentId: componentId ? `${componentId}.close` : 'codegen_design-system_src_design-system_sidebar_sidebar.tsx_427',
                    size: "small",
                    icon: /*#__PURE__*/ jsx(CloseIcon, {}),
                    "aria-label": "Close",
                    onClick: ()=>{
                        contentContext.setIsClosed();
                    }
                })
            }) : null
        ]
    });
}
function PanelHeaderTitle({ title, dangerouslyAppendEmotionCSS }) {
    return /*#__PURE__*/ jsx("div", {
        title: title,
        css: [
            {
                alignSelf: 'center',
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis'
            },
            dangerouslyAppendEmotionCSS
        ],
        children: title
    });
}
function PanelHeaderButtons({ children, dangerouslyAppendEmotionCSS }) {
    const { theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsx("div", {
        css: [
            {
                display: 'flex',
                alignItems: 'center',
                gap: theme.spacing.xs,
                paddingRight: theme.spacing.xs
            },
            dangerouslyAppendEmotionCSS
        ],
        children: children
    });
}
function PanelBody({ children, dangerouslyAppendEmotionCSS }) {
    const { theme } = useDesignSystemTheme();
    const [shouldBeFocusable, setShouldBeFocusable] = useState(false);
    const bodyRef = useRef(null);
    useEffect(()=>{
        const ref = bodyRef.current;
        if (ref) {
            if (ref.scrollHeight > ref.clientHeight) {
                setShouldBeFocusable(true);
            } else {
                setShouldBeFocusable(false);
            }
        }
    }, []);
    return /*#__PURE__*/ jsx("div", {
        ref: bodyRef,
        // Needed to make panel body content focusable when scrollable for keyboard-only users to be able to focus & scroll
        // eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex
        tabIndex: shouldBeFocusable ? 0 : -1,
        css: [
            {
                height: '100%',
                overflowX: 'hidden',
                overflowY: 'auto',
                padding: '0 8px',
                colorScheme: theme.isDarkMode ? 'dark' : 'light'
            },
            dangerouslyAppendEmotionCSS
        ],
        children: children
    });
}
const Sidebar = /* #__PURE__ */ (()=>{
    function Sidebar({ position, children, dangerouslyAppendEmotionCSS, ...dataProps }) {
        const { theme } = useDesignSystemTheme();
        const value = useMemo(()=>{
            return {
                position: position || 'left'
            };
        }, [
            position
        ]);
        return /*#__PURE__*/ jsx(SidebarContext.Provider, {
            value: value,
            children: /*#__PURE__*/ jsx("div", {
                ...addDebugOutlineIfEnabled(),
                ...dataProps,
                css: [
                    {
                        display: 'flex',
                        height: '100%',
                        backgroundColor: theme.colors.backgroundPrimary,
                        flexDirection: position === 'right' ? 'row-reverse' : 'row',
                        borderStyle: 'solid',
                        borderColor: theme.colors.border,
                        borderWidth: position === 'right' ? `0 0 0 ${theme.general.borderWidth}px` : `0px ${theme.general.borderWidth}px 0 0`,
                        boxSizing: 'content-box',
                        position: 'relative'
                    },
                    dangerouslyAppendEmotionCSS
                ],
                children: children
            })
        });
    }
    Sidebar.Content = Content$1;
    Sidebar.Nav = Nav;
    Sidebar.NavButton = NavButton;
    Sidebar.Panel = Panel;
    Sidebar.PanelHeader = PanelHeader;
    Sidebar.PanelHeaderTitle = PanelHeaderTitle;
    Sidebar.PanelHeaderButtons = PanelHeaderButtons;
    Sidebar.PanelBody = PanelBody;
    return Sidebar;
})();

function Stepper({ direction: requestedDirection, currentStepIndex: currentStep, steps, localizeStepNumber, responsive = true, onStepClicked }) {
    const { theme } = useDesignSystemTheme();
    const ref = useRef(null);
    const { direction } = useResponsiveDirection({
        ref,
        requestedDirection,
        responsive,
        enabled: steps.length > 0
    });
    if (steps.length === 0) {
        return null;
    }
    const isHorizontal = direction === 'horizontal';
    const currentStepIndex = currentStep ? Math.min(steps.length - 1, Math.max(0, currentStep)) : 0;
    return /*#__PURE__*/ jsx("ol", {
        ...addDebugOutlineIfEnabled(),
        css: /*#__PURE__*/ css(getStepsStyle(theme, isHorizontal)),
        ref: ref,
        children: steps.map((step, index)=>{
            const isCurrentStep = index === currentStepIndex;
            const isLastStep = index === steps.length - 1;
            const displayEndingDivider = !isLastStep;
            const contentTitleLevel = 4;
            const StepIcon = step.icon;
            const clickEnabled = step.clickEnabled && onStepClicked;
            const { icon, iconBackgroundColor, iconTextColor, titleTextColor, titleAsParagraph, descriptionTextColor, hasStepItemIconBorder, stepItemIconBorderColor } = getStepContentStyleFields(theme, isCurrentStep, step.status, StepIcon, step.additionalVerticalContent);
            return /*#__PURE__*/ jsx("li", {
                "aria-current": isCurrentStep,
                css: css(getStepItemStyle(theme, isHorizontal, isLastStep)),
                ...step.status === 'error' && {
                    'data-error': true
                },
                ...step.status === 'loading' && {
                    'data-loading': true
                },
                children: /*#__PURE__*/ jsxs(StepContentGrid, {
                    isHorizontal: isHorizontal,
                    children: [
                        /*#__PURE__*/ jsx("div", {
                            css: css(getStepItemIconParentStyle(theme, iconBackgroundColor, hasStepItemIconBorder, stepItemIconBorderColor, Boolean(clickEnabled))),
                            onClick: clickEnabled ? ()=>onStepClicked(index) : undefined,
                            children: StepIcon ? /*#__PURE__*/ jsx(StepIcon, {
                                statusColor: iconTextColor,
                                status: step.status
                            }) : icon ? /*#__PURE__*/ jsx("span", {
                                css: {
                                    color: iconTextColor,
                                    display: 'flex'
                                },
                                children: icon
                            }) : /*#__PURE__*/ jsx(Typography.Title, {
                                level: contentTitleLevel,
                                css: {
                                    color: `${iconTextColor} !important`
                                },
                                withoutMargins: true,
                                children: localizeStepNumber(index + 1)
                            })
                        }),
                        /*#__PURE__*/ jsx("span", {
                            onClick: clickEnabled ? ()=>onStepClicked(index) : undefined,
                            onKeyDown: (event)=>{
                                if (event.key === 'Enter' && clickEnabled) {
                                    onStepClicked(index);
                                }
                            },
                            css: {
                                cursor: clickEnabled ? 'pointer' : undefined
                            },
                            tabIndex: clickEnabled ? 0 : undefined,
                            role: clickEnabled ? 'button' : undefined,
                            children: titleAsParagraph ? /*#__PURE__*/ jsx(Typography.Text, {
                                withoutMargins: true,
                                css: {
                                    flexShrink: 0,
                                    color: `${titleTextColor} !important`
                                },
                                children: step.title
                            }) : /*#__PURE__*/ jsx(Typography.Title, {
                                level: 4,
                                withoutMargins: true,
                                css: {
                                    flexShrink: 0,
                                    color: `${titleTextColor} !important`
                                },
                                children: step.title
                            })
                        }),
                        displayEndingDivider && /*#__PURE__*/ jsx("div", {
                            css: css(getStepEndingDividerStyles(theme, isHorizontal, step.status === 'completed'))
                        }),
                        (step.description || step.additionalVerticalContent && !isHorizontal) && /*#__PURE__*/ jsxs("div", {
                            css: css(getStepDescriptionStyles(theme, isHorizontal, isLastStep, descriptionTextColor)),
                            children: [
                                step.description && /*#__PURE__*/ jsx(Typography.Text, {
                                    css: css(getStepDescriptionTextStyles(descriptionTextColor)),
                                    withoutMargins: true,
                                    size: "sm",
                                    children: step.description
                                }),
                                step.additionalVerticalContent && !isHorizontal && /*#__PURE__*/ jsx("div", {
                                    css: css(getAdditionalVerticalStepContentStyles(theme, Boolean(step.description))),
                                    children: step.additionalVerticalContent
                                })
                            ]
                        })
                    ]
                })
            }, index);
        })
    });
}
function getStepsStyle(theme, isHorizontal) {
    return /*#__PURE__*/ css({
        listStyle: 'none',
        display: 'flex',
        flexDirection: isHorizontal ? 'row' : 'column',
        flexWrap: 'wrap',
        alignItems: 'flex-start',
        gap: isHorizontal ? theme.spacing.sm : theme.spacing.xs,
        width: '100%',
        margin: '0',
        padding: '0'
    });
}
function getStepItemStyle(theme, isHorizontal, isLastStep) {
    return /*#__PURE__*/ css({
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'flex-start',
        flexGrow: isLastStep ? 0 : 1,
        marginRight: isLastStep && isHorizontal ? theme.spacing.lg + theme.spacing.md : 0,
        width: isHorizontal ? undefined : '100%'
    });
}
function getStepContentStyleFields(theme, isCurrentStep, status, icon, additionalVerticalContent) {
    const fields = getStepContentStyleFieldsFromStatus(theme, isCurrentStep, status, !isUndefined(additionalVerticalContent));
    if (icon) {
        return {
            ...fields,
            icon: undefined,
            iconBackgroundColor: undefined,
            iconTextColor: getCustomIconColor(theme, isCurrentStep, status),
            hasStepItemIconBorder: false
        };
    }
    return fields;
}
function getCustomIconColor(theme, isCurrentStep, status) {
    switch(status){
        case 'completed':
            return theme.colors.actionLinkDefault;
        case 'loading':
            return theme.colors.textPlaceholder;
        case 'error':
            return theme.colors.textValidationDanger;
        case 'warning':
            return theme.colors.textValidationWarning;
        default:
        case 'upcoming':
            return isCurrentStep ? theme.colors.actionLinkDefault : theme.colors.textPlaceholder;
    }
}
function getStepContentStyleFieldsFromStatus(theme, isCurrentStep, status, hasAdditionalVerticalContent) {
    switch(status){
        case 'completed':
            return {
                icon: /*#__PURE__*/ jsx(CheckIcon, {}),
                iconBackgroundColor: isCurrentStep ? theme.colors.actionLinkDefault : theme.isDarkMode ? theme.colors.blue800 : theme.colors.blue100,
                iconTextColor: isCurrentStep ? theme.colors.white : theme.colors.actionLinkDefault,
                titleAsParagraph: false,
                titleTextColor: theme.colors.actionLinkDefault,
                descriptionTextColor: theme.colors.textSecondary,
                hasStepItemIconBorder: true,
                stepItemIconBorderColor: theme.colors.actionDefaultBackgroundPress
            };
        case 'loading':
            return {
                icon: /*#__PURE__*/ jsx(LoadingIcon, {
                    spin: true,
                    css: {
                        fontSize: isCurrentStep ? theme.typography.fontSizeXl : theme.typography.fontSizeLg
                    }
                }),
                iconBackgroundColor: undefined,
                iconTextColor: theme.colors.textPlaceholder,
                titleAsParagraph: false,
                titleTextColor: isCurrentStep ? theme.colors.textPrimary : theme.colors.textSecondary,
                descriptionTextColor: theme.colors.textSecondary,
                hasStepItemIconBorder: false
            };
        case 'error':
            return {
                icon: /*#__PURE__*/ jsx(DangerIcon, {}),
                iconBackgroundColor: isCurrentStep ? theme.colors.textValidationDanger : theme.colors.backgroundDanger,
                iconTextColor: isCurrentStep ? theme.colors.white : theme.colors.textValidationDanger,
                titleAsParagraph: false,
                titleTextColor: theme.colors.textValidationDanger,
                descriptionTextColor: theme.colors.textValidationDanger,
                hasStepItemIconBorder: true,
                stepItemIconBorderColor: theme.colors.borderDanger
            };
        case 'warning':
            return {
                icon: /*#__PURE__*/ jsx(WarningIcon, {}),
                iconBackgroundColor: isCurrentStep ? theme.colors.textValidationWarning : theme.colors.backgroundWarning,
                iconTextColor: isCurrentStep ? theme.colors.white : theme.colors.textValidationWarning,
                titleAsParagraph: false,
                titleTextColor: theme.colors.textValidationWarning,
                descriptionTextColor: theme.colors.textValidationWarning,
                hasStepItemIconBorder: true,
                stepItemIconBorderColor: theme.colors.borderWarning
            };
        default:
        case 'upcoming':
            if (isCurrentStep) {
                return {
                    icon: undefined,
                    iconBackgroundColor: theme.colors.actionLinkDefault,
                    iconTextColor: 'white',
                    titleAsParagraph: false,
                    titleTextColor: theme.colors.actionLinkDefault,
                    descriptionTextColor: hasAdditionalVerticalContent ? theme.colors.textSecondary : theme.colors.textPrimary,
                    hasStepItemIconBorder: false
                };
            }
            return {
                icon: undefined,
                iconBackgroundColor: undefined,
                iconTextColor: theme.colors.textPlaceholder,
                titleAsParagraph: true,
                titleTextColor: theme.colors.textSecondary,
                descriptionTextColor: theme.colors.textSecondary,
                hasStepItemIconBorder: true,
                stepItemIconBorderColor: theme.colors.border
            };
    }
}
const MaxHorizontalStepDescriptionWidth = 140;
const StepIconSize = DEFAULT_SPACING_UNIT * 4;
function getStepItemIconParentStyle(theme, iconBackgroundColor, hasStepItemIconBorder, stepItemIconBorderColor, clickEnabled) {
    return /*#__PURE__*/ css({
        width: StepIconSize,
        height: StepIconSize,
        backgroundColor: iconBackgroundColor,
        borderRadius: theme.borders.borderRadiusFull,
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        fontSize: '20px',
        flexShrink: 0,
        border: hasStepItemIconBorder ? `1px solid ${stepItemIconBorderColor ?? theme.colors.textPlaceholder}` : undefined,
        boxSizing: 'border-box',
        cursor: clickEnabled ? 'pointer' : undefined
    });
}
function getStepEndingDividerStyles(theme, isHorizontal, isCompleted) {
    const backgroundColor = isCompleted ? theme.colors.actionLinkDefault : theme.colors.border;
    if (isHorizontal) {
        return /*#__PURE__*/ css({
            backgroundColor,
            height: '1px',
            width: '100%',
            minWidth: theme.spacing.md
        });
    }
    return /*#__PURE__*/ css({
        backgroundColor,
        height: '100%',
        minHeight: theme.spacing.lg,
        width: '1px',
        alignSelf: 'flex-start',
        marginLeft: theme.spacing.md
    });
}
function getStepDescriptionStyles(theme, isHorizontal, isLastStep, textColor) {
    return /*#__PURE__*/ css({
        alignSelf: 'flex-start',
        width: '100%',
        gridColumn: isHorizontal || isLastStep ? '2 / span 2' : undefined,
        maxWidth: isHorizontal ? MaxHorizontalStepDescriptionWidth : undefined,
        paddingBottom: isHorizontal ? undefined : theme.spacing.sm,
        color: `${textColor} !important`
    });
}
function getStepDescriptionTextStyles(textColor) {
    return /*#__PURE__*/ css({
        color: `${textColor} !important`
    });
}
function getAdditionalVerticalStepContentStyles(theme, addTopPadding) {
    return /*#__PURE__*/ css({
        paddingTop: addTopPadding ? theme.spacing.sm : 0
    });
}
function StepContentGrid({ children, isHorizontal }) {
    const { theme } = useDesignSystemTheme();
    if (isHorizontal) {
        return /*#__PURE__*/ jsx("div", {
            css: {
                display: 'grid',
                gridTemplateColumns: `${StepIconSize}px fit-content(100%) 1fr`,
                gridTemplateRows: `${StepIconSize}px auto`,
                alignItems: 'center',
                justifyItems: 'flex-start',
                gridColumnGap: theme.spacing.sm,
                width: '100%'
            },
            children: children
        });
    }
    return /*#__PURE__*/ jsx("div", {
        css: {
            display: 'grid',
            gridTemplateColumns: `${StepIconSize}px minmax(0, 1fr)`,
            alignItems: 'center',
            justifyItems: 'flex-start',
            gridColumnGap: theme.spacing.sm,
            gridRowGap: theme.spacing.xs,
            width: '100%',
            '& > :first-child': {
                // horizontally center the first column (circle/icon)
                justifySelf: 'center'
            }
        },
        children: children
    });
}
// Ant design uses the same value for their stepper and to works well for us as well.
const MinimumHorizonalDirectionWidth = 532;
// exported for unit test
function useResponsiveDirection({ requestedDirection = 'horizontal', responsive, enabled, ref }) {
    const [direction, setDirection] = useState(requestedDirection);
    useEffect(()=>{
        if (requestedDirection === 'vertical' || !enabled || !responsive || !ref.current) {
            return;
        }
        let timeoutId;
        const resizeObserver = new ResizeObserver((entries)=>{
            timeoutId = requestAnimationFrame(()=>{
                if (entries.length === 1) {
                    const width = entries[0].target.clientWidth || 0;
                    setDirection(width < MinimumHorizonalDirectionWidth ? 'vertical' : 'horizontal');
                }
            });
        });
        if (ref.current) {
            resizeObserver.observe(ref.current);
        }
        return ()=>{
            resizeObserver.disconnect();
            cancelAnimationFrame(timeoutId);
        };
    }, [
        requestedDirection,
        enabled,
        ref,
        responsive
    ]);
    return {
        direction
    };
}

function useStepperStepsFromWizardSteps(wizardSteps, currentStepIdx, hideDescriptionForFutureSteps) {
    return useMemo(()=>wizardSteps.map((wizardStep, stepIdx)=>({
                ...pick(wizardStep, [
                    'title',
                    'status'
                ]),
                description: hideDescriptionForFutureSteps && !(stepIdx <= currentStepIdx || wizardStep.status === 'completed' || wizardStep.status === 'error' || wizardStep.status === 'warning') ? undefined : wizardStep.description,
                additionalVerticalContent: wizardStep.additionalHorizontalLayoutStepContent,
                clickEnabled: isUndefined(wizardStep.clickEnabled) ? isWizardStepEnabled(wizardSteps, stepIdx, currentStepIdx, wizardStep.status) : wizardStep.clickEnabled
            })), [
        currentStepIdx,
        hideDescriptionForFutureSteps,
        wizardSteps
    ]);
}
function isWizardStepEnabled(steps, stepIdx, currentStepIdx, status) {
    if (stepIdx < currentStepIdx || status === 'completed' || status === 'error' || status === 'warning') {
        return true;
    }
    // if every step before stepIdx is completed then the step is enabled
    return steps.slice(0, stepIdx).every((step)=>step.status === 'completed');
}

function HorizontalWizardStepsContent({ steps: wizardSteps, currentStepIndex, localizeStepNumber, enableClickingToSteps, goToStep, hideDescriptionForFutureSteps = false }) {
    const { theme } = useDesignSystemTheme();
    const stepperSteps = useStepperStepsFromWizardSteps(wizardSteps, currentStepIndex, hideDescriptionForFutureSteps);
    const expandContentToFullHeight = wizardSteps[currentStepIndex].expandContentToFullHeight ?? true;
    const disableDefaultScrollBehavior = wizardSteps[currentStepIndex].disableDefaultScrollBehavior ?? false;
    return /*#__PURE__*/ jsxs(Fragment, {
        children: [
            /*#__PURE__*/ jsx(Stepper, {
                currentStepIndex: currentStepIndex,
                direction: "horizontal",
                localizeStepNumber: localizeStepNumber,
                steps: stepperSteps,
                responsive: false,
                onStepClicked: enableClickingToSteps ? goToStep : undefined
            }),
            /*#__PURE__*/ jsx("div", {
                css: {
                    marginTop: theme.spacing.md,
                    flexGrow: expandContentToFullHeight ? 1 : undefined,
                    overflowY: disableDefaultScrollBehavior ? 'hidden' : 'auto',
                    ...!disableDefaultScrollBehavior ? getShadowScrollStyles(theme) : {}
                },
                children: wizardSteps[currentStepIndex].content
            })
        ]
    });
}

// Buttons are returned in order with primary button last
function getWizardFooterButtons({ goToNextStepOrDone, isLastStep, currentStepIndex, goToPreviousStep, busyValidatingNextStep, nextButtonDisabled, nextButtonLoading, nextButtonContentOverride, previousButtonContentOverride, previousStepButtonHidden, previousButtonDisabled, previousButtonLoading, cancelButtonContent, cancelStepButtonHidden, nextButtonContent, previousButtonContent, doneButtonContent, extraFooterButtonsLeft, extraFooterButtonsRight, onCancel, moveCancelToOtherSide, componentId, tooltipContent }) {
    return compact([
        !cancelStepButtonHidden && (moveCancelToOtherSide ? /*#__PURE__*/ jsx("div", {
            css: {
                flexGrow: 1
            },
            children: /*#__PURE__*/ jsx(CancelButton, {
                onCancel: onCancel,
                cancelButtonContent: cancelButtonContent,
                // eslint-disable-next-line @databricks/no-dynamic-property-value -- Validated through JSX componentId exempt:4b31acca-dae8-44a7-a7f5-fa3e1bc167c5
                componentId: componentId ? `${componentId}.cancel` : undefined
            })
        }, "cancel") : /*#__PURE__*/ jsx(CancelButton, {
            onCancel: onCancel,
            cancelButtonContent: cancelButtonContent,
            // eslint-disable-next-line @databricks/no-dynamic-property-value -- Validated through JSX componentId exempt:acae06b0-d74d-44ee-86db-15b3014e2bcf
            componentId: componentId ? `${componentId}.cancel` : undefined
        }, "cancel")),
        currentStepIndex > 0 && !previousStepButtonHidden && /*#__PURE__*/ jsx(Button, {
            onClick: goToPreviousStep,
            type: "tertiary",
            disabled: previousButtonDisabled,
            loading: previousButtonLoading,
            componentId: componentId ? `${componentId}.previous` : 'dubois-wizard-footer-previous',
            children: previousButtonContentOverride ? previousButtonContentOverride : previousButtonContent
        }, "previous"),
        extraFooterButtonsLeft && extraFooterButtonsLeft.map((buttonProps, index)=>/*#__PURE__*/ createElement(ButtonWithTooltip, {
                ...buttonProps,
                key: `extra-left-${index}`
            })),
        /*#__PURE__*/ jsx(ButtonWithTooltip, {
            onClick: goToNextStepOrDone,
            disabled: nextButtonDisabled,
            tooltipContent: tooltipContent,
            loading: nextButtonLoading || busyValidatingNextStep,
            type: (extraFooterButtonsRight?.length ?? 0) > 0 ? undefined : 'primary',
            componentId: componentId ? `${componentId}.next` : 'dubois-wizard-footer-next',
            children: nextButtonContentOverride ? nextButtonContentOverride : isLastStep ? doneButtonContent : nextButtonContent
        }, "next"),
        extraFooterButtonsRight && extraFooterButtonsRight.map((buttonProps, index)=>/*#__PURE__*/ createElement(ButtonWithTooltip, {
                ...buttonProps,
                type: index === extraFooterButtonsRight.length - 1 ? 'primary' : undefined,
                key: `extra-right-${index}`
            }))
    ]);
}
function CancelButton({ onCancel, cancelButtonContent, componentId }) {
    return /*#__PURE__*/ jsx(Button, {
        onClick: onCancel,
        type: "tertiary",
        componentId: componentId ?? 'dubois-wizard-footer-cancel',
        children: cancelButtonContent
    }, "cancel");
}
function ButtonWithTooltip({ tooltipContent, disabled, children, ...props }) {
    return tooltipContent ? /*#__PURE__*/ jsx(Tooltip, {
        componentId: "dubois-wizard-footer-tooltip",
        content: tooltipContent,
        children: /*#__PURE__*/ jsx(Button, {
            ...props,
            disabled: disabled,
            children: children
        })
    }) : /*#__PURE__*/ jsx(Button, {
        ...props,
        disabled: disabled,
        children: children
    });
}

function HorizontalWizardContent({ width, height, steps, currentStepIndex, localizeStepNumber, onStepsChange, enableClickingToSteps, hideDescriptionForFutureSteps, ...footerProps }) {
    return /*#__PURE__*/ jsxs("div", {
        css: {
            width,
            height,
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'flex-start'
        },
        ...addDebugOutlineIfEnabled(),
        children: [
            /*#__PURE__*/ jsx(HorizontalWizardStepsContent, {
                steps: steps,
                currentStepIndex: currentStepIndex,
                localizeStepNumber: localizeStepNumber,
                enableClickingToSteps: Boolean(enableClickingToSteps),
                goToStep: footerProps.goToStep,
                hideDescriptionForFutureSteps: hideDescriptionForFutureSteps
            }),
            /*#__PURE__*/ jsx(Spacer, {
                size: "lg"
            }),
            /*#__PURE__*/ jsx(WizardFooter, {
                currentStepIndex: currentStepIndex,
                ...steps[currentStepIndex],
                ...footerProps,
                moveCancelToOtherSide: true
            })
        ]
    });
}
function WizardFooter(props) {
    const { theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsx("div", {
        css: {
            display: 'flex',
            flexDirection: 'row',
            justifyContent: 'flex-end',
            columnGap: theme.spacing.sm,
            paddingTop: theme.spacing.md,
            paddingBottom: theme.spacing.md,
            borderTop: `1px solid ${theme.colors.border}`
        },
        children: getWizardFooterButtons(props)
    });
}

function Root({ children, initialContentId }) {
    const [currentContentId, setCurrentContentId] = useState(initialContentId);
    return /*#__PURE__*/ jsx(DocumentationSideBarContext.Provider, {
        value: useMemo(()=>({
                currentContentId,
                setCurrentContentId
            }), [
            currentContentId,
            setCurrentContentId
        ]),
        children: children
    });
}
const DocumentationSideBarContext = /*#__PURE__*/ React__default.createContext({
    currentContentId: undefined,
    setCurrentContentId: noop
});
const useDocumentationSidebarContext = ()=>{
    const context = React__default.useContext(DocumentationSideBarContext);
    return context;
};
function Trigger({ contentId, label, tooltipContent, asChild, children, ...tooltipProps }) {
    const { theme } = useDesignSystemTheme();
    const { setCurrentContentId } = useDocumentationSidebarContext();
    const triggerProps = useMemo(()=>({
            onClick: ()=>setCurrentContentId(contentId),
            [`aria-label`]: label
        }), [
        contentId,
        label,
        setCurrentContentId
    ]);
    const renderAsChild = asChild && /*#__PURE__*/ React__default.isValidElement(children);
    return /*#__PURE__*/ jsx(Tooltip, {
        ...tooltipProps,
        content: tooltipContent,
        children: renderAsChild ? /*#__PURE__*/ React__default.cloneElement(children, triggerProps) : /*#__PURE__*/ jsx("button", {
            css: {
                border: 'none',
                backgroundColor: 'transparent',
                padding: 0,
                display: 'flex',
                height: 'var(--spacing-md)',
                alignItems: 'center',
                cursor: 'pointer'
            },
            ...triggerProps,
            children: /*#__PURE__*/ jsx(InfoSmallIcon, {
                css: {
                    fontSize: theme.typography.fontSizeSm,
                    color: theme.colors.textSecondary
                }
            })
        })
    });
}
function Content({ title, modalTitleWhenCompact, width, children, closeLabel, displayModalWhenCompact }) {
    const { theme } = useDesignSystemTheme();
    const { currentContentId, setCurrentContentId } = useDocumentationSidebarContext();
    if (isUndefined(currentContentId)) {
        return null;
    }
    const content = /*#__PURE__*/ React__default.isValidElement(children) ? /*#__PURE__*/ React__default.cloneElement(children, {
        contentId: currentContentId
    }) : children;
    if (displayModalWhenCompact) {
        return /*#__PURE__*/ jsx(Modal, {
            componentId: "documentation-side-bar-compact-modal",
            visible: true,
            size: "wide",
            onOk: ()=>setCurrentContentId(undefined),
            okText: closeLabel,
            okButtonProps: {
                type: undefined
            },
            onCancel: ()=>setCurrentContentId(undefined),
            title: modalTitleWhenCompact ?? title,
            children: content
        });
    }
    return /*#__PURE__*/ jsx(Sidebar, {
        position: "right",
        dangerouslyAppendEmotionCSS: {
            border: 'none'
        },
        children: /*#__PURE__*/ jsx(Sidebar.Content, {
            componentId: "documentation-side-bar-content",
            openPanelId: 0,
            closable: true,
            disableResize: true,
            enableCompact: true,
            width: width,
            children: /*#__PURE__*/ jsx(Sidebar.Panel, {
                panelId: 0,
                children: /*#__PURE__*/ jsxs("div", {
                    css: {
                        height: '100%',
                        display: 'flex',
                        flexDirection: 'column',
                        rowGap: theme.spacing.md,
                        borderRadius: theme.legacyBorders.borderRadiusLg,
                        border: `1px solid ${theme.colors.backgroundSecondary}`,
                        padding: `${theme.spacing.md}px ${theme.spacing.lg}px`,
                        backgroundColor: theme.colors.backgroundSecondary
                    },
                    children: [
                        /*#__PURE__*/ jsxs("div", {
                            css: {
                                display: 'flex',
                                flexDirection: 'row',
                                justifyContent: 'space-between',
                                alignItems: 'center',
                                width: '100%'
                            },
                            children: [
                                /*#__PURE__*/ jsx(Typography.Text, {
                                    color: "secondary",
                                    children: title
                                }),
                                /*#__PURE__*/ jsx(Button, {
                                    "aria-label": closeLabel,
                                    icon: /*#__PURE__*/ jsx(CloseIcon, {}),
                                    componentId: "documentation-side-bar-close",
                                    onClick: ()=>setCurrentContentId(undefined)
                                })
                            ]
                        }),
                        content
                    ]
                })
            })
        })
    });
}

var DocumentationSidebar = /*#__PURE__*/Object.freeze({
  __proto__: null,
  Content: Content,
  Root: Root,
  Trigger: Trigger,
  useDocumentationSidebarContext: useDocumentationSidebarContext
});

const SMALL_FIXED_VERTICAL_STEPPER_WIDTH = 240;
const FIXED_VERTICAL_STEPPER_WIDTH = 280;
const MAX_VERTICAL_WIZARD_CONTENT_WIDTH = 920;
const DOCUMENTATION_SIDEBAR_WIDTH = 400;
const EXTRA_COMPACT_BUTTON_HEIGHT = 32 + 24; // button height + gap
function VerticalWizardContent({ width, height, steps: wizardSteps, currentStepIndex, localizeStepNumber, onStepsChange, title, padding, verticalCompactButtonContent, enableClickingToSteps, verticalDocumentationSidebarConfig, hideDescriptionForFutureSteps = false, contentMaxWidth, compactStepperBreakpoint, ...footerProps }) {
    const { theme } = useDesignSystemTheme();
    const stepperSteps = useStepperStepsFromWizardSteps(wizardSteps, currentStepIndex, hideDescriptionForFutureSteps);
    const expandContentToFullHeight = wizardSteps[currentStepIndex].expandContentToFullHeight ?? true;
    const disableDefaultScrollBehavior = wizardSteps[currentStepIndex].disableDefaultScrollBehavior ?? false;
    const displayDocumentationSideBar = Boolean(verticalDocumentationSidebarConfig);
    const Wrapper = displayDocumentationSideBar ? Root : React__default.Fragment;
    const displayCompactStepper = useMediaQuery({
        query: `(max-width: ${compactStepperBreakpoint ?? theme.responsive.breakpoints.lg}px)`
    }) && Boolean(verticalCompactButtonContent);
    const displayCompactSidebar = useMediaQuery({
        query: `(max-width: ${theme.responsive.breakpoints.xxl}px)`
    });
    return /*#__PURE__*/ jsx(Wrapper, {
        ...displayDocumentationSideBar && {
            initialContentId: verticalDocumentationSidebarConfig?.initialContentId
        },
        children: /*#__PURE__*/ jsxs("div", {
            css: {
                width,
                height: expandContentToFullHeight ? height : 'fit-content',
                maxHeight: '100%',
                display: 'flex',
                flexDirection: displayCompactStepper ? 'column' : 'row',
                gap: theme.spacing.lg,
                justifyContent: 'center'
            },
            ...addDebugOutlineIfEnabled(),
            children: [
                !displayCompactStepper && /*#__PURE__*/ jsxs("div", {
                    css: {
                        display: 'flex',
                        flexDirection: 'column',
                        flexShrink: 0,
                        rowGap: theme.spacing.lg,
                        paddingTop: theme.spacing.lg,
                        paddingBottom: theme.spacing.lg,
                        height: 'fit-content',
                        width: SMALL_FIXED_VERTICAL_STEPPER_WIDTH,
                        [compactStepperBreakpoint ? `@media (max-width: ${compactStepperBreakpoint}px)` : `@media (min-width: ${theme.responsive.breakpoints.xl}px)`]: {
                            width: FIXED_VERTICAL_STEPPER_WIDTH
                        },
                        overflowX: 'hidden'
                    },
                    children: [
                        title,
                        /*#__PURE__*/ jsx(Stepper, {
                            currentStepIndex: currentStepIndex,
                            direction: "vertical",
                            localizeStepNumber: localizeStepNumber,
                            steps: stepperSteps,
                            responsive: false,
                            onStepClicked: enableClickingToSteps ? footerProps.goToStep : undefined
                        })
                    ]
                }),
                displayCompactStepper && /*#__PURE__*/ jsxs(Root$1, {
                    componentId: "codegen_design-system_src_~patterns_wizard_verticalwizardcontent.tsx_93",
                    children: [
                        /*#__PURE__*/ jsx(Trigger$1, {
                            asChild: true,
                            children: /*#__PURE__*/ jsx("div", {
                                children: /*#__PURE__*/ jsx(Button, {
                                    icon: /*#__PURE__*/ jsx(ListIcon, {}),
                                    componentId: "dubois-wizard-vertical-compact-show-stepper-popover",
                                    children: verticalCompactButtonContent?.(currentStepIndex, stepperSteps.length)
                                })
                            })
                        }),
                        /*#__PURE__*/ jsx(Content$2, {
                            align: "start",
                            side: "bottom",
                            css: {
                                padding: theme.spacing.md
                            },
                            children: /*#__PURE__*/ jsx(Stepper, {
                                currentStepIndex: currentStepIndex,
                                direction: "vertical",
                                localizeStepNumber: localizeStepNumber,
                                steps: stepperSteps,
                                responsive: false,
                                onStepClicked: enableClickingToSteps ? footerProps.goToStep : undefined
                            })
                        })
                    ]
                }),
                /*#__PURE__*/ jsxs("div", {
                    css: {
                        display: 'flex',
                        flexDirection: 'column',
                        columnGap: theme.spacing.lg,
                        border: `1px solid ${theme.colors.border}`,
                        borderRadius: theme.legacyBorders.borderRadiusLg,
                        flexGrow: 1,
                        padding: padding ?? theme.spacing.lg,
                        height: displayCompactStepper ? `calc(100% - ${EXTRA_COMPACT_BUTTON_HEIGHT}px)` : '100%',
                        maxWidth: contentMaxWidth ?? MAX_VERTICAL_WIZARD_CONTENT_WIDTH
                    },
                    children: [
                        /*#__PURE__*/ jsx("div", {
                            css: {
                                flexGrow: expandContentToFullHeight ? 1 : undefined,
                                overflowY: disableDefaultScrollBehavior ? 'hidden' : 'auto',
                                ...!disableDefaultScrollBehavior ? getShadowScrollStyles(theme) : {},
                                borderRadius: theme.legacyBorders.borderRadiusLg
                            },
                            children: wizardSteps[currentStepIndex].content
                        }),
                        /*#__PURE__*/ jsx("div", {
                            css: {
                                display: 'flex',
                                flexDirection: 'row',
                                justifyContent: 'flex-end',
                                columnGap: theme.spacing.sm,
                                ...padding !== undefined && {
                                    padding: theme.spacing.lg
                                },
                                paddingTop: theme.spacing.md
                            },
                            children: getWizardFooterButtons({
                                currentStepIndex: currentStepIndex,
                                ...wizardSteps[currentStepIndex],
                                ...footerProps,
                                moveCancelToOtherSide: true
                            })
                        })
                    ]
                }),
                displayDocumentationSideBar && verticalDocumentationSidebarConfig && /*#__PURE__*/ jsx(Content, {
                    width: displayCompactSidebar ? undefined : DOCUMENTATION_SIDEBAR_WIDTH,
                    title: verticalDocumentationSidebarConfig.title,
                    modalTitleWhenCompact: verticalDocumentationSidebarConfig.modalTitleWhenCompact,
                    closeLabel: verticalDocumentationSidebarConfig.closeLabel,
                    displayModalWhenCompact: displayCompactSidebar,
                    children: verticalDocumentationSidebarConfig.content
                })
            ]
        })
    });
}

function useWizardCurrentStep({ currentStepIndex, setCurrentStepIndex, totalSteps, onValidateStepChange, onStepChanged }) {
    const [busyValidatingNextStep, setBusyValidatingNextStep] = useState(false);
    const isLastStep = useMemo(()=>currentStepIndex === totalSteps - 1, [
        currentStepIndex,
        totalSteps
    ]);
    const onStepsChange = useCallback(async (step, completed = false)=>{
        if (!completed && step === currentStepIndex) return;
        setCurrentStepIndex(step);
        onStepChanged({
            step,
            completed
        });
    }, [
        currentStepIndex,
        onStepChanged,
        setCurrentStepIndex
    ]);
    const goToNextStepOrDone = useCallback(async ()=>{
        if (onValidateStepChange) {
            setBusyValidatingNextStep(true);
            try {
                const approvedStepChange = await onValidateStepChange(currentStepIndex);
                if (!approvedStepChange) {
                    return;
                }
            } finally{
                setBusyValidatingNextStep(false);
            }
        }
        onStepsChange(Math.min(currentStepIndex + 1, totalSteps - 1), isLastStep);
    }, [
        currentStepIndex,
        isLastStep,
        onStepsChange,
        onValidateStepChange,
        totalSteps
    ]);
    const goToPreviousStep = useCallback(()=>{
        if (currentStepIndex > 0) {
            onStepsChange(currentStepIndex - 1);
        }
    }, [
        currentStepIndex,
        onStepsChange
    ]);
    const goToStep = useCallback((step)=>{
        if (step > -1 && step < totalSteps) {
            onStepsChange(step);
        }
    }, [
        onStepsChange,
        totalSteps
    ]);
    return {
        busyValidatingNextStep,
        isLastStep,
        onStepsChange,
        goToNextStepOrDone,
        goToPreviousStep,
        goToStep
    };
}

function Wizard({ steps, onStepChanged, onValidateStepChange, initialStep, ...props }) {
    const [currentStepIndex, setCurrentStepIndex] = useState(initialStep ?? 0);
    const currentStepProps = useWizardCurrentStep({
        currentStepIndex,
        setCurrentStepIndex,
        totalSteps: steps.length,
        onStepChanged,
        onValidateStepChange
    });
    return /*#__PURE__*/ jsx(WizardControlled, {
        ...currentStepProps,
        currentStepIndex: currentStepIndex,
        initialStep: initialStep,
        steps: steps,
        onStepChanged: onStepChanged,
        ...props
    });
}
function WizardControlled({ initialStep = 0, layout = 'vertical', width = '100%', height = '100%', steps, title, ...restOfProps }) {
    if (steps.length === 0 || !isUndefined(initialStep) && (initialStep < 0 || initialStep >= steps.length)) {
        return null;
    }
    if (layout === 'vertical') {
        return /*#__PURE__*/ jsx(VerticalWizardContent, {
            width: width,
            height: height,
            steps: steps,
            title: title,
            ...restOfProps
        });
    }
    return /*#__PURE__*/ jsx(HorizontalWizardContent, {
        width: width,
        height: height,
        steps: steps,
        ...restOfProps
    });
}

function WizardModal({ onStepChanged, onCancel, initialStep, steps, onModalClose, localizeStepNumber, cancelButtonContent, nextButtonContent, previousButtonContent, doneButtonContent, enableClickingToSteps, ...modalProps }) {
    const [currentStepIndex, setCurrentStepIndex] = useState(initialStep ?? 0);
    const { onStepsChange, isLastStep, ...footerActions } = useWizardCurrentStep({
        currentStepIndex,
        setCurrentStepIndex,
        totalSteps: steps.length,
        onStepChanged
    });
    if (steps.length === 0 || !isUndefined(initialStep) && (initialStep < 0 || initialStep >= steps.length)) {
        return null;
    }
    const footerButtons = getWizardFooterButtons({
        onCancel,
        isLastStep,
        currentStepIndex,
        doneButtonContent,
        previousButtonContent,
        nextButtonContent,
        cancelButtonContent,
        ...footerActions,
        ...steps[currentStepIndex]
    });
    return /*#__PURE__*/ jsx(Modal, {
        ...modalProps,
        onCancel: onModalClose,
        size: "wide",
        footer: footerButtons,
        children: /*#__PURE__*/ jsx(HorizontalWizardStepsContent, {
            steps: steps,
            currentStepIndex: currentStepIndex,
            localizeStepNumber: localizeStepNumber,
            enableClickingToSteps: Boolean(enableClickingToSteps),
            goToStep: footerActions.goToStep
        })
    });
}

function WizardStepContentWrapper({ header, title, description, alertContent, children }) {
    const { theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsxs("div", {
        css: {
            display: 'flex',
            flexDirection: 'column',
            height: '100%'
        },
        children: [
            /*#__PURE__*/ jsxs("div", {
                style: {
                    backgroundColor: theme.colors.backgroundSecondary,
                    padding: theme.spacing.lg,
                    display: 'flex',
                    flexDirection: 'column',
                    borderTopLeftRadius: theme.legacyBorders.borderRadiusLg,
                    borderTopRightRadius: theme.legacyBorders.borderRadiusLg
                },
                children: [
                    /*#__PURE__*/ jsx(Typography.Text, {
                        size: "sm",
                        style: {
                            fontWeight: 500
                        },
                        children: header
                    }),
                    /*#__PURE__*/ jsx(Typography.Title, {
                        withoutMargins: true,
                        style: {
                            paddingTop: theme.spacing.lg
                        },
                        level: 3,
                        children: title
                    }),
                    /*#__PURE__*/ jsx(Typography.Text, {
                        color: "secondary",
                        children: description
                    })
                ]
            }),
            alertContent && /*#__PURE__*/ jsx("div", {
                css: {
                    padding: `${theme.spacing.lg}px ${theme.spacing.lg}px 0`
                },
                children: alertContent
            }),
            /*#__PURE__*/ jsx("div", {
                css: {
                    display: 'flex',
                    flexDirection: 'column',
                    height: '100%',
                    padding: `${theme.spacing.lg}px ${theme.spacing.lg}px 0`,
                    overflowY: 'auto',
                    ...getShadowScrollStyles(theme)
                },
                children: children
            })
        ]
    });
}

export { Content$1 as C, DocumentationSidebar as D, FIXED_VERTICAL_STEPPER_WIDTH as F, MAX_VERTICAL_WIZARD_CONTENT_WIDTH as M, Nav as N, Panel as P, Spacer as S, Wizard as W, WizardControlled as a, WizardModal as b, WizardStepContentWrapper as c, NavButton as d, PanelHeader as e, PanelHeaderTitle as f, PanelHeaderButtons as g, PanelBody as h, Sidebar as i, Stepper as j, useWizardCurrentStep as u };
//# sourceMappingURL=WizardStepContentWrapper-BRXcq4zt.js.map
