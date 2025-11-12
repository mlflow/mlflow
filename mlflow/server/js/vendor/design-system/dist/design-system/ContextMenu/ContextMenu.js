import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { useMergeRefs } from '@floating-ui/react';
import { ContextMenuArrow, ContextMenuCheckboxItem, ContextMenuContent, ContextMenuGroup, ContextMenuItem, ContextMenuItemIndicator, ContextMenuLabel, ContextMenuPortal, ContextMenuRadioGroup, ContextMenuRadioItem, ContextMenuSeparator, ContextMenuSub, ContextMenuSubContent, ContextMenuSubTrigger, ContextMenuTrigger, ContextMenu as RadixContextMenu, } from '@radix-ui/react-context-menu';
import React, { createContext, useCallback, useMemo, useRef } from 'react';
import { CheckIcon, ChevronRightIcon, DesignSystemEventProviderAnalyticsEventTypes, DesignSystemEventProviderComponentTypes, useDesignSystemEventComponentCallbacks, useDesignSystemSafexFlags, useDesignSystemTheme, useModalContext, } from '..';
import { dropdownContentStyles, dropdownItemStyles, dropdownSeparatorStyles } from '../DropdownMenu/DropdownMenu';
import { handleKeyboardNavigation } from '../DropdownMenu/utils';
import { useDesignSystemContext } from '../Hooks/useDesignSystemContext';
import { getNewChildren } from '../_shared_/Menu';
import { useNotifyOnFirstView } from '../utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
import { safex } from '../utils/safex';
export const Trigger = ContextMenuTrigger;
export const ItemIndicator = ContextMenuItemIndicator;
export const Group = ContextMenuGroup;
export const Arrow = ContextMenuArrow;
export const Sub = ContextMenuSub;
const ContextMenuProps = createContext({ isOpen: false, setIsOpen: (isOpen) => { } });
const useContextMenuProps = () => React.useContext(ContextMenuProps);
export const Root = ({ children, onOpenChange, ...props }) => {
    const [isOpen, setIsOpen] = React.useState(false);
    const handleChange = (isOpen) => {
        setIsOpen(isOpen);
        onOpenChange?.(isOpen);
    };
    return (_jsx(RadixContextMenu, { onOpenChange: handleChange, ...props, children: _jsx(ContextMenuProps.Provider, { value: { isOpen, setIsOpen }, children: children }) }));
};
export const SubTrigger = ({ children, disabledReason, withChevron, ...props }) => {
    const { theme } = useDesignSystemTheme();
    const ref = useRef(null);
    return (_jsxs(ContextMenuSubTrigger, { ...props, css: dropdownItemStyles(theme), ref: ref, onKeyDown: (e) => {
            if (e.key === 'Tab' || e.key === 'ArrowDown' || e.key === 'ArrowUp') {
                e.preventDefault();
            }
            props.onKeyDown?.(e);
        }, children: [getNewChildren(children, props, disabledReason, ref), withChevron && (_jsx(ContextMenu.Hint, { children: _jsx(ChevronRightIcon, {}) }))] }));
};
export const Content = ({ children, minWidth, forceCloseOnEscape, onEscapeKeyDown, onKeyDown, ...childrenProps }) => {
    const { getPopupContainer } = useDesignSystemContext();
    const { theme } = useDesignSystemTheme();
    const { isInsideModal } = useModalContext();
    const { isOpen, setIsOpen } = useContextMenuProps();
    const { useNewShadows, useNewBorderColors } = useDesignSystemSafexFlags();
    return (_jsx(ContextMenuPortal, { container: getPopupContainer && getPopupContainer(), children: isOpen && (_jsx(ContextMenuContent, { ...addDebugOutlineIfEnabled(), onWheel: (e) => {
                e.stopPropagation();
            }, onTouchMove: (e) => {
                e.stopPropagation();
            }, onKeyDown: (e) => {
                // This is a workaround for Radix's ContextMenu.Content not receiving Escape key events
                // when nested inside a modal. We need to stop propagation of the event so that the modal
                // doesn't close when the DropdownMenu should.
                if (e.key === 'Escape') {
                    if (isInsideModal || forceCloseOnEscape) {
                        e.stopPropagation();
                        setIsOpen(false);
                    }
                    onEscapeKeyDown?.(e.nativeEvent);
                }
                else if (e.key === 'Tab' || e.key === 'ArrowDown' || e.key === 'ArrowUp') {
                    e.stopPropagation();
                    handleKeyboardNavigation(e);
                }
                onKeyDown?.(e);
            }, ...childrenProps, css: [dropdownContentStyles(theme, useNewShadows, useNewBorderColors), { minWidth }], children: children })) }));
};
export const SubContent = ({ children, minWidth, ...childrenProps }) => {
    const { getPopupContainer } = useDesignSystemContext();
    const { theme } = useDesignSystemTheme();
    const { useNewShadows, useNewBorderColors } = useDesignSystemSafexFlags();
    return (_jsx(ContextMenuPortal, { container: getPopupContainer && getPopupContainer(), children: _jsx(ContextMenuSubContent, { ...addDebugOutlineIfEnabled(), ...childrenProps, onWheel: (e) => {
                e.stopPropagation();
            }, onTouchMove: (e) => {
                e.stopPropagation();
            }, css: [dropdownContentStyles(theme, useNewShadows, useNewBorderColors), { minWidth }], onKeyDown: (e) => {
                if (e.key === 'Tab' || e.key === 'ArrowDown' || e.key === 'ArrowUp') {
                    e.stopPropagation();
                    handleKeyboardNavigation(e);
                }
                childrenProps.onKeyDown?.(e);
            }, children: children }) }));
};
export const Item = ({ children, disabledReason, onClick, componentId, analyticsEvents, asChild, ...props }) => {
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.contextMenu', false);
    const { theme } = useDesignSystemTheme();
    const ref = useRef(null);
    const memoizedAnalyticsEvents = useMemo(() => analyticsEvents ??
        (emitOnView
            ? [DesignSystemEventProviderAnalyticsEventTypes.OnClick, DesignSystemEventProviderAnalyticsEventTypes.OnView]
            : [DesignSystemEventProviderAnalyticsEventTypes.OnClick]), [analyticsEvents, emitOnView]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.ContextMenuItem,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
    });
    const { elementRef: contextMenuItemRef } = useNotifyOnFirstView({
        onView: !asChild ? eventContext.onView : () => { },
    });
    const mergedRef = useMergeRefs([ref, contextMenuItemRef]);
    const onClickWrapper = useCallback((e) => {
        if (!asChild) {
            eventContext.onClick(e);
        }
        onClick?.(e);
    }, [asChild, eventContext, onClick]);
    return (_jsx(ContextMenuItem, { ...props, asChild: asChild, onClick: onClickWrapper, css: dropdownItemStyles(theme), ref: mergedRef, onKeyDown: (e) => {
            if (e.key === 'Tab' || e.key === 'ArrowDown' || e.key === 'ArrowUp') {
                e.preventDefault();
            }
            props.onKeyDown?.(e);
        }, ...eventContext.dataComponentProps, children: getNewChildren(children, props, disabledReason, ref) }));
};
export const CheckboxItem = ({ children, disabledReason, onCheckedChange, componentId, analyticsEvents, ...props }) => {
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.contextMenu', false);
    const { theme } = useDesignSystemTheme();
    const ref = useRef(null);
    const memoizedAnalyticsEvents = useMemo(() => analyticsEvents ??
        (emitOnView
            ? [
                DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
                DesignSystemEventProviderAnalyticsEventTypes.OnView,
            ]
            : [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange]), [analyticsEvents, emitOnView]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.ContextMenuCheckboxItem,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        valueHasNoPii: true,
    });
    const { elementRef: contextMenuCheckboxItemRef } = useNotifyOnFirstView({
        onView: eventContext.onView,
        value: props.checked ?? props.defaultChecked,
    });
    const mergedRef = useMergeRefs([ref, contextMenuCheckboxItemRef]);
    const onCheckedChangeWrapper = useCallback((checked) => {
        eventContext.onValueChange(checked);
        onCheckedChange?.(checked);
    }, [eventContext, onCheckedChange]);
    return (_jsxs(ContextMenuCheckboxItem, { ...props, onCheckedChange: onCheckedChangeWrapper, css: dropdownItemStyles(theme), ref: mergedRef, onKeyDown: (e) => {
            if (e.key === 'Tab' || e.key === 'ArrowDown' || e.key === 'ArrowUp') {
                e.preventDefault();
            }
            props.onKeyDown?.(e);
        }, ...eventContext.dataComponentProps, children: [_jsx(ContextMenuItemIndicator, { css: itemIndicatorStyles(theme), children: _jsx(CheckIcon, {}) }), !props.checked && _jsx("div", { style: { width: theme.general.iconFontSize + theme.spacing.xs } }), getNewChildren(children, props, disabledReason, ref)] }));
};
export const RadioGroup = ({ onValueChange, componentId, analyticsEvents, valueHasNoPii, ...props }) => {
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.contextMenu', false);
    const memoizedAnalyticsEvents = useMemo(() => analyticsEvents ??
        (emitOnView
            ? [
                DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
                DesignSystemEventProviderAnalyticsEventTypes.OnView,
            ]
            : [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange]), [analyticsEvents, emitOnView]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.ContextMenuRadioGroup,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        valueHasNoPii,
    });
    const { elementRef: contextMenuRadioGroupRef } = useNotifyOnFirstView({
        onView: eventContext.onView,
        value: props.value ?? props.defaultValue,
    });
    const onValueChangeWrapper = useCallback((value) => {
        eventContext.onValueChange(value);
        onValueChange?.(value);
    }, [eventContext, onValueChange]);
    return (_jsx(ContextMenuRadioGroup, { ref: contextMenuRadioGroupRef, ...props, onValueChange: onValueChangeWrapper, ...eventContext.dataComponentProps }));
};
export const RadioItem = ({ children, disabledReason, ...props }) => {
    const { theme } = useDesignSystemTheme();
    const ref = useRef(null);
    return (_jsxs(ContextMenuRadioItem, { ...props, css: [
            dropdownItemStyles(theme),
            {
                '&[data-state="unchecked"]': {
                    paddingLeft: theme.general.iconFontSize + theme.spacing.xs + theme.spacing.sm,
                },
            },
        ], ref: ref, children: [_jsx(ContextMenuItemIndicator, { css: itemIndicatorStyles(theme), children: _jsx(CheckIcon, {}) }), getNewChildren(children, props, disabledReason, ref)] }));
};
export const Label = ({ children, ...props }) => {
    const { theme } = useDesignSystemTheme();
    return (_jsx(ContextMenuLabel, { ...props, css: { color: theme.colors.textSecondary, padding: `${theme.spacing.sm - 2}px ${theme.spacing.sm}px` }, children: children }));
};
export const Hint = ({ children }) => {
    const { theme } = useDesignSystemTheme();
    return _jsx("span", { css: { display: 'inline-flex', marginLeft: 'auto', paddingLeft: theme.spacing.sm }, children: children });
};
export const Separator = () => {
    const { theme } = useDesignSystemTheme();
    return _jsx(ContextMenuSeparator, { css: dropdownSeparatorStyles(theme) });
};
export const itemIndicatorStyles = (theme) => css({ display: 'inline-flex', paddingRight: theme.spacing.xs });
export const ContextMenu = {
    Root,
    Trigger,
    Label,
    Item,
    Group,
    RadioGroup,
    CheckboxItem,
    RadioItem,
    Arrow,
    Separator,
    Sub,
    SubTrigger,
    SubContent,
    Content,
    Hint,
};
//# sourceMappingURL=ContextMenu.js.map