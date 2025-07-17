import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { useMergeRefs } from '@floating-ui/react';
import * as ScrollArea from '@radix-ui/react-scroll-area';
import * as RadixTabs from '@radix-ui/react-tabs';
import { debounce } from 'lodash';
import React, { useMemo } from 'react';
import { Button, CloseSmallIcon, PlusIcon, getShadowScrollStyles, DesignSystemEventProviderAnalyticsEventTypes, DesignSystemEventProviderComponentTypes, useDesignSystemEventComponentCallbacks, useDesignSystemTheme, useNotifyOnFirstView, } from '..';
import { getCommonTabsListStyles, getCommonTabsTriggerStyles } from '../_shared_';
import { safex } from '../utils/safex';
const TabsRootContext = React.createContext({
    activeValue: undefined,
    dataComponentProps: {
        'data-component-id': 'design_system.tabs.default_component_id',
        'data-component-type': DesignSystemEventProviderComponentTypes.Tabs,
    },
});
const TabsListContext = React.createContext({ viewportRef: { current: null } });
export const Root = React.forwardRef(({ value, defaultValue, onValueChange, componentId, analyticsEvents, valueHasNoPii, ...props }, forwardedRef) => {
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.tabs', false);
    const isControlled = value !== undefined;
    const [uncontrolledActiveValue, setUncontrolledActiveValue] = React.useState(defaultValue);
    const memoizedAnalyticsEvents = useMemo(() => analyticsEvents ??
        (emitOnView
            ? [
                DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
                DesignSystemEventProviderAnalyticsEventTypes.OnView,
            ]
            : [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange]), [analyticsEvents, emitOnView]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.Tabs,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        valueHasNoPii,
        shouldStartInteraction: true,
    });
    const { elementRef: tabsRootRef } = useNotifyOnFirstView({
        onView: eventContext.onView,
        value: value ?? defaultValue,
    });
    const mergedRef = useMergeRefs([forwardedRef, tabsRootRef]);
    const onValueChangeWrapper = (value) => {
        eventContext.onValueChange(value);
        if (onValueChange) {
            onValueChange(value);
        }
        if (!isControlled) {
            setUncontrolledActiveValue(value);
        }
    };
    return (_jsx(TabsRootContext.Provider, { value: {
            activeValue: isControlled ? value : uncontrolledActiveValue,
            dataComponentProps: eventContext.dataComponentProps,
        }, children: _jsx(RadixTabs.Root, { value: value, defaultValue: defaultValue, onValueChange: onValueChangeWrapper, ...props, ref: mergedRef }) }));
});
export const List = React.forwardRef(({ addButtonProps, scrollAreaViewportCss, tabListCss, children, dangerouslyAppendEmotionCSS, shadowScrollStylesBackgroundColor, scrollbarHeight, getScrollAreaViewportRef, ...props }, forwardedRef) => {
    const viewportRef = React.useRef(null);
    const { dataComponentProps } = React.useContext(TabsRootContext);
    const css = useListStyles(shadowScrollStylesBackgroundColor, scrollbarHeight);
    React.useEffect(() => {
        if (getScrollAreaViewportRef) {
            getScrollAreaViewportRef(viewportRef.current);
        }
    }, [getScrollAreaViewportRef]);
    return (_jsx(TabsListContext.Provider, { value: { viewportRef }, children: _jsxs("div", { css: [css['container'], dangerouslyAppendEmotionCSS], children: [_jsxs(ScrollArea.Root, { type: "hover", css: [css['root']], children: [_jsx(ScrollArea.Viewport, { css: [
                                css['viewport'],
                                scrollAreaViewportCss,
                                {
                                    // Added to prevent adding and removing tabs from leaving extra empty spaces between existing tabs and the "+" button
                                    '& > div': { display: 'inline-block !important' },
                                },
                            ], ref: viewportRef, children: _jsx(RadixTabs.List, { css: [css['list'], tabListCss], ...props, ref: forwardedRef, ...dataComponentProps, children: children }) }), _jsx(ScrollArea.Scrollbar, { orientation: "horizontal", css: css['scrollbar'], children: _jsx(ScrollArea.Thumb, { css: css['thumb'] }) })] }), addButtonProps && (_jsx("div", { css: [css['addButtonContainer'], addButtonProps.dangerouslyAppendEmotionCSS], children: _jsx(Button, { icon: _jsx(PlusIcon, {}), size: "small", "aria-label": "Add tab", css: css['addButton'], onClick: addButtonProps.onClick, componentId: `${dataComponentProps['data-component-id']}.add_tab`, className: addButtonProps.className }) }))] }) }));
});
export const Trigger = React.forwardRef(({ onClose, value, disabled, children, ...props }, forwardedRef) => {
    const triggerRef = React.useRef(null);
    const mergedRef = useMergeRefs([forwardedRef, triggerRef]);
    const { activeValue, dataComponentProps } = React.useContext(TabsRootContext);
    const componentId = dataComponentProps['data-component-id'];
    const { viewportRef } = React.useContext(TabsListContext);
    const isClosable = onClose !== undefined && !disabled;
    const css = useTriggerStyles(isClosable);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.Button,
        componentId: `${componentId}.close_tab`,
        analyticsEvents: [DesignSystemEventProviderAnalyticsEventTypes.OnClick],
    });
    const scrollActiveTabIntoView = React.useCallback(() => {
        if (triggerRef.current && viewportRef.current && activeValue === value) {
            const viewportPosition = viewportRef.current.getBoundingClientRect();
            const triggerPosition = triggerRef.current.getBoundingClientRect();
            if (triggerPosition.left < viewportPosition.left) {
                viewportRef.current.scrollLeft -= viewportPosition.left - triggerPosition.left;
            }
            else if (triggerPosition.right > viewportPosition.right) {
                viewportRef.current.scrollLeft += triggerPosition.right - viewportPosition.right;
            }
        }
    }, [viewportRef, activeValue, value]);
    const debouncedScrollActiveTabIntoView = React.useMemo(() => debounce(scrollActiveTabIntoView, 10), [scrollActiveTabIntoView]);
    React.useEffect(() => {
        scrollActiveTabIntoView();
    }, [scrollActiveTabIntoView]);
    React.useEffect(() => {
        if (!viewportRef.current || !triggerRef.current) {
            return;
        }
        const resizeObserver = new ResizeObserver(debouncedScrollActiveTabIntoView);
        resizeObserver.observe(viewportRef.current);
        resizeObserver.observe(triggerRef.current);
        return () => {
            resizeObserver.disconnect();
            debouncedScrollActiveTabIntoView.cancel();
        };
    }, [debouncedScrollActiveTabIntoView, viewportRef]);
    return (_jsxs(RadixTabs.Trigger, { css: css['trigger'], value: value, disabled: disabled, 
        // The close icon cannot be focused within the trigger button
        // Instead, we close the tab when the Delete key is pressed
        onKeyDown: (e) => {
            if (isClosable && e.key === 'Delete') {
                eventContext.onClick(e);
                e.stopPropagation();
                e.preventDefault();
                onClose(value);
            }
        }, 
        // Middle click also closes the tab
        // The Radix Tabs implementation uses onMouseDown for handling clicking tabs so we use it here as well
        onMouseDown: (e) => {
            if (isClosable && e.button === 1) {
                eventContext.onClick(e);
                e.stopPropagation();
                e.preventDefault();
                onClose(value);
            }
        }, ...props, ref: mergedRef, children: [children, isClosable && (
            // An icon is used instead of a button to prevent nesting a button within a button
            _jsx(CloseSmallIcon, { onMouseDown: (e) => {
                    // The Radix Tabs implementation only allows the trigger to be selected when the left mouse
                    // button is clicked and not when the control key is pressed (to avoid MacOS right click).
                    // Reimplementing the same behavior for the close icon in the trigger
                    if (!disabled && e.button === 0 && e.ctrlKey === false) {
                        eventContext.onClick(e);
                        // Clicking the close icon should not select the tab
                        e.stopPropagation();
                        e.preventDefault();
                        onClose(value);
                    }
                }, css: css['closeSmallIcon'], "aria-hidden": "false", "aria-label": "Press delete to close the tab" }))] }));
});
export const Content = React.forwardRef(({ ...props }, forwardedRef) => {
    const { theme } = useDesignSystemTheme();
    const css = useContentStyles(theme);
    return _jsx(RadixTabs.Content, { css: css, ...props, ref: forwardedRef });
});
const useListStyles = (shadowScrollStylesBackgroundColor, scrollbarHeight) => {
    const { theme } = useDesignSystemTheme();
    const containerStyles = getCommonTabsListStyles(theme);
    return {
        container: containerStyles,
        root: { overflow: 'hidden' },
        viewport: {
            ...getShadowScrollStyles(theme, {
                orientation: 'horizontal',
                backgroundColor: shadowScrollStylesBackgroundColor,
            }),
        },
        list: { display: 'flex', alignItems: 'center' },
        scrollbar: {
            display: 'flex',
            flexDirection: 'column',
            userSelect: 'none',
            /* Disable browser handling of all panning and zooming gestures on touch devices */
            touchAction: 'none',
            height: scrollbarHeight ?? 3,
        },
        thumb: {
            flex: 1,
            background: theme.isDarkMode ? 'rgba(255, 255, 255, 0.2)' : 'rgba(17, 23, 28, 0.2)',
            '&:hover': {
                background: theme.isDarkMode ? 'rgba(255, 255, 255, 0.3)' : 'rgba(17, 23, 28, 0.3)',
            },
            borderRadius: theme.borders.borderRadiusSm,
            position: 'relative',
        },
        addButtonContainer: { flex: 1 },
        addButton: { margin: '2px 0 6px 0' },
    };
};
const useTriggerStyles = (isClosable) => {
    const { theme } = useDesignSystemTheme();
    const commonTriggerStyles = getCommonTabsTriggerStyles(theme);
    return {
        trigger: {
            ...commonTriggerStyles,
            alignItems: 'center',
            justifyContent: isClosable ? 'space-between' : 'center',
            minWidth: isClosable ? theme.spacing.lg + theme.spacing.md : theme.spacing.lg,
            color: theme.colors.textSecondary,
            lineHeight: theme.typography.lineHeightBase,
            whiteSpace: 'nowrap',
            border: 'none',
            padding: `${theme.spacing.xs}px 0 ${theme.spacing.sm}px 0`,
            // The close icon is hidden on inactive tabs until the tab is hovered
            // Checking for the last icon to handle cases where the tab name includes an icon
            [`& > .anticon:last-of-type`]: {
                visibility: 'hidden',
            },
            '&:hover': {
                cursor: 'pointer',
                color: theme.colors.actionDefaultTextHover,
                [`& > .anticon:last-of-type`]: {
                    visibility: 'visible',
                },
            },
            '&:active': {
                color: theme.colors.actionDefaultTextPress,
            },
            outlineStyle: 'none',
            outlineColor: theme.colors.actionDefaultBorderFocus,
            '&:focus-visible': {
                outlineStyle: 'auto',
            },
            '&[data-state="active"]': {
                color: theme.colors.textPrimary,
                // Use box-shadow instead of border to prevent it from affecting the size of the element, which results in visual
                // jumping when switching tabs.
                boxShadow: `inset 0 -4px 0 ${theme.colors.actionPrimaryBackgroundDefault}`,
                // The close icon is always visible on active tabs
                [`& > .anticon:last-of-type`]: {
                    visibility: 'visible',
                },
            },
            '&[data-disabled]': {
                color: theme.colors.actionDisabledText,
                '&:hover': {
                    cursor: 'not-allowed',
                },
            },
        },
        closeSmallIcon: {
            marginLeft: theme.spacing.xs,
            color: theme.colors.textSecondary,
            '&:hover': {
                color: theme.colors.actionDefaultTextHover,
            },
            '&:active': {
                color: theme.colors.actionDefaultTextPress,
            },
        },
    };
};
const useContentStyles = (theme) => {
    // This is needed so force mounted content is not displayed when the tab is inactive
    return {
        color: theme.colors.textPrimary,
        '&[data-state="inactive"]': {
            display: 'none',
        },
    };
};
//# sourceMappingURL=Tabs.js.map