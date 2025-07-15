import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { useMergeRefs } from '@floating-ui/react';
import * as DropdownMenu from '@radix-ui/react-dropdown-menu';
import React, { createContext, forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef } from 'react';
import { handleKeyboardNavigation } from './utils';
import { useFormContext } from '../../development/Form/Form';
import { DesignSystemEventProviderAnalyticsEventTypes, DesignSystemEventProviderComponentTypes, useDesignSystemEventComponentCallbacks, } from '../DesignSystemEventProvider/DesignSystemEventProvider';
import { useDesignSystemTheme } from '../Hooks';
import { useDesignSystemContext } from '../Hooks/useDesignSystemContext';
import { CheckIcon, ChevronRightIcon } from '../Icon';
import { useModalContext } from '../Modal';
import { getNewChildren } from '../_shared_/Menu';
import { useDesignSystemSafexFlags, useNotifyOnFirstView } from '../utils';
import { getDarkModePortalStyles, importantify } from '../utils/css-utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
import { safex } from '../utils/safex';
const DropdownContext = createContext({ isOpen: false, setIsOpen: (isOpen) => { } });
const useDropdownContext = () => React.useContext(DropdownContext);
export const Root = ({ children, itemHtmlType, ...props }) => {
    const [isOpen, setIsOpen] = React.useState(Boolean(props.defaultOpen || props.open));
    const useExternalState = useRef(props.open !== undefined || props.onOpenChange !== undefined).current;
    useEffect(() => {
        if (useExternalState) {
            setIsOpen(Boolean(props.open));
        }
    }, [useExternalState, props.open]);
    const handleOpenChange = (isOpen) => {
        if (!useExternalState) {
            setIsOpen(isOpen);
        }
        // In case the consumer doesn't manage open state but wants to listen to the callback
        if (props.onOpenChange) {
            props.onOpenChange(isOpen);
        }
    };
    return (_jsx(DropdownMenu.Root, { ...props, ...(!useExternalState && {
            open: isOpen,
            onOpenChange: handleOpenChange,
        }), children: _jsx(DropdownContext.Provider, { value: {
                isOpen: useExternalState ? props.open : isOpen,
                setIsOpen: useExternalState ? props.onOpenChange : handleOpenChange,
                itemHtmlType,
            }, children: children }) }));
};
export const Content = forwardRef(function Content({ children, minWidth = 220, matchTriggerWidth, forceCloseOnEscape, onEscapeKeyDown, onKeyDown, ...props }, ref) {
    const { getPopupContainer } = useDesignSystemContext();
    const { theme } = useDesignSystemTheme();
    const { useNewShadows, useNewBorderColors } = useDesignSystemSafexFlags();
    const { setIsOpen } = useDropdownContext();
    const { isInsideModal } = useModalContext();
    return (_jsx(DropdownMenu.Portal, { container: getPopupContainer && getPopupContainer(), children: _jsx(DropdownMenu.Content, { ...addDebugOutlineIfEnabled(), ref: ref, loop: true, css: [
                contentStyles(theme, useNewShadows, useNewBorderColors),
                { minWidth },
                matchTriggerWidth ? { width: 'var(--radix-dropdown-menu-trigger-width)' } : {},
            ], sideOffset: 4, align: "start", onKeyDown: (e) => {
                // This is a workaround for Radix's DropdownMenu.Content not receiving Escape key events
                // when nested inside a modal. We need to stop propagation of the event so that the modal
                // doesn't close when the DropdownMenu should.
                if (e.key === 'Escape') {
                    if (isInsideModal || forceCloseOnEscape) {
                        e.stopPropagation();
                        setIsOpen?.(false);
                    }
                    onEscapeKeyDown?.(e.nativeEvent);
                }
                if (e.key === 'Tab' || e.key === 'ArrowDown' || e.key === 'ArrowUp') {
                    handleKeyboardNavigation(e);
                }
                onKeyDown?.(e);
            }, ...props, onWheel: (e) => {
                e.stopPropagation();
                props?.onWheel?.(e);
            }, onTouchMove: (e) => {
                e.stopPropagation();
                props?.onTouchMove?.(e);
            }, children: children }) }));
});
export const SubContent = forwardRef(function Content({ children, minWidth = 220, onKeyDown, ...props }, ref) {
    const { getPopupContainer } = useDesignSystemContext();
    const { theme } = useDesignSystemTheme();
    const { useNewShadows, useNewBorderColors } = useDesignSystemSafexFlags();
    const [contentFitsInViewport, setContentFitsInViewport] = React.useState(true);
    const [dataSide, setDataSide] = React.useState(null);
    const { isOpen } = useSubContext();
    const elemRef = useRef(null);
    useImperativeHandle(ref, () => elemRef.current);
    const checkAvailableWidth = useCallback(() => {
        if (elemRef.current) {
            const elemStyle = getComputedStyle(elemRef.current);
            const availableWidth = parseFloat(elemStyle.getPropertyValue('--radix-dropdown-menu-content-available-width'));
            const elemWidth = elemRef.current.offsetWidth;
            const openOnSide = elemRef.current.getAttribute('data-side');
            if (openOnSide === 'left' || openOnSide === 'right') {
                setDataSide(openOnSide);
            }
            else {
                setDataSide(null);
            }
            if (availableWidth < elemWidth) {
                setContentFitsInViewport(false);
            }
            else {
                setContentFitsInViewport(true);
            }
        }
    }, []);
    useEffect(() => {
        window.addEventListener('resize', checkAvailableWidth);
        checkAvailableWidth();
        return () => {
            window.removeEventListener('resize', checkAvailableWidth);
        };
    }, [checkAvailableWidth]);
    useEffect(() => {
        if (isOpen) {
            setTimeout(() => {
                checkAvailableWidth();
            }, 25);
        }
    }, [isOpen, checkAvailableWidth]);
    let transformCalc = `calc(var(--radix-dropdown-menu-content-available-width) + var(--radix-dropdown-menu-trigger-width) * -1)`;
    if (dataSide === 'left') {
        transformCalc = `calc(var(--radix-dropdown-menu-trigger-width) - var(--radix-dropdown-menu-content-available-width))`;
    }
    const responsiveCss = `
    transform-origin: var(--radix-dropdown-menu-content-transform-origin) !important;
    transform: translateX(${transformCalc}) !important;
`;
    return (_jsx(DropdownMenu.Portal, { container: getPopupContainer && getPopupContainer(), children: _jsx(DropdownMenu.SubContent, { ...addDebugOutlineIfEnabled(), ref: elemRef, loop: true, css: [
                contentStyles(theme, useNewShadows, useNewBorderColors),
                { minWidth },
                contentFitsInViewport ? '' : responsiveCss,
            ], sideOffset: -2, alignOffset: -5, onKeyDown: (e) => {
                if (e.key === 'Tab' || e.key === 'ArrowDown' || e.key === 'ArrowUp') {
                    e.stopPropagation();
                    handleKeyboardNavigation(e);
                }
                onKeyDown?.(e);
            }, ...props, children: children }) }));
});
export const Trigger = forwardRef(function Trigger({ children, ...props }, ref) {
    return (_jsx(DropdownMenu.Trigger, { ...addDebugOutlineIfEnabled(), ref: ref, ...props, children: children }));
});
export const Item = forwardRef(function Item({ children, disabledReason, danger, onClick, componentId, analyticsEvents, ...props }, ref) {
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.dropdownMenu', false);
    const formContext = useFormContext();
    const { itemHtmlType } = useDropdownContext();
    const itemRef = useRef(null);
    useImperativeHandle(ref, () => itemRef.current);
    const memoizedAnalyticsEvents = useMemo(() => analyticsEvents ??
        (emitOnView
            ? [DesignSystemEventProviderAnalyticsEventTypes.OnClick, DesignSystemEventProviderAnalyticsEventTypes.OnView]
            : [DesignSystemEventProviderAnalyticsEventTypes.OnClick]), [analyticsEvents, emitOnView]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.DropdownMenuItem,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        // If the item is a submit item and is part of a form, it is not the subject of the interaction, the form submission is
        isInteractionSubject: !(itemHtmlType === 'submit' && formContext.componentId),
    });
    const { elementRef: dropdownMenuItemRef } = useNotifyOnFirstView({
        onView: !props.asChild ? eventContext.onView : () => { },
    });
    const mergedRefs = useMergeRefs([itemRef, dropdownMenuItemRef]);
    return (_jsx(DropdownMenu.Item, { css: (theme) => [dropdownItemStyles, danger && dangerItemStyles(theme)], ref: mergedRefs, onClick: (e) => {
            if (props.disabled) {
                e.preventDefault();
            }
            else {
                if (!props.asChild) {
                    eventContext.onClick(e);
                }
                if (itemHtmlType === 'submit' && formContext.formRef?.current) {
                    e.preventDefault();
                    formContext.formRef.current.requestSubmit();
                }
                onClick?.(e);
            }
        }, onKeyDown: (e) => {
            if (e.key === 'Tab' || e.key === 'ArrowDown' || e.key === 'ArrowUp') {
                e.preventDefault();
            }
            props.onKeyDown?.(e);
        }, ...props, ...eventContext.dataComponentProps, children: getNewChildren(children, props, disabledReason, itemRef) }));
});
export const Label = forwardRef(function Label({ children, ...props }, ref) {
    return (_jsx(DropdownMenu.Label, { ref: ref, css: [
            dropdownItemStyles,
            (theme) => ({
                color: theme.colors.textSecondary,
                '&:hover': {
                    cursor: 'default',
                },
            }),
        ], ...props, children: children }));
});
export const Separator = forwardRef(function Separator({ children, ...props }, ref) {
    return (_jsx(DropdownMenu.Separator, { ref: ref, css: dropdownSeparatorStyles, ...props, children: children }));
});
export const SubTrigger = forwardRef(function TriggerItem({ children, disabledReason, ...props }, ref) {
    const subTriggerRef = useRef(null);
    useImperativeHandle(ref, () => subTriggerRef.current);
    return (_jsxs(DropdownMenu.SubTrigger, { ref: subTriggerRef, css: [
            dropdownItemStyles,
            (theme) => ({
                '&[data-state="open"]': {
                    backgroundColor: theme.colors.actionTertiaryBackgroundHover,
                },
            }),
        ], onKeyDown: (e) => {
            if (e.key === 'Tab' || e.key === 'ArrowDown' || e.key === 'ArrowUp') {
                e.preventDefault();
            }
            props.onKeyDown?.(e);
        }, ...props, children: [getNewChildren(children, props, disabledReason, subTriggerRef), _jsx(HintColumn, { css: (theme) => ({
                    margin: CONSTANTS.subMenuIconMargin(theme),
                    display: 'flex',
                    alignSelf: 'stretch',
                    alignItems: 'center',
                }), children: _jsx(ChevronRightIcon, { css: (theme) => ({ fontSize: CONSTANTS.subMenuIconSize(theme) }) }) })] }));
});
/**
 * Deprecated. Use `SubTrigger` instead.
 * @deprecated
 */
export const TriggerItem = SubTrigger;
export const CheckboxItem = forwardRef(function CheckboxItem({ children, disabledReason, componentId, analyticsEvents, onCheckedChange, ...props }, ref) {
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.dropdownMenu', false);
    const checkboxItemRef = useRef(null);
    useImperativeHandle(ref, () => checkboxItemRef.current);
    const memoizedAnalyticsEvents = useMemo(() => analyticsEvents ??
        (emitOnView
            ? [
                DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
                DesignSystemEventProviderAnalyticsEventTypes.OnView,
            ]
            : [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange]), [analyticsEvents, emitOnView]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.DropdownMenuCheckboxItem,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        valueHasNoPii: true,
    });
    const { elementRef: checkboxItemOnViewRef } = useNotifyOnFirstView({
        onView: eventContext.onView,
        value: props.checked ?? props.defaultChecked,
    });
    const mergedRefs = useMergeRefs([checkboxItemRef, checkboxItemOnViewRef]);
    const onCheckedChangeWrapper = useCallback((checked) => {
        eventContext.onValueChange(checked);
        onCheckedChange?.(checked);
    }, [eventContext, onCheckedChange]);
    return (_jsx(DropdownMenu.CheckboxItem, { ref: mergedRefs, css: (theme) => [dropdownItemStyles, checkboxItemStyles(theme)], onCheckedChange: onCheckedChangeWrapper, onKeyDown: (e) => {
            if (e.key === 'Tab' || e.key === 'ArrowDown' || e.key === 'ArrowUp') {
                e.preventDefault();
            }
            props.onKeyDown?.(e);
        }, ...props, ...eventContext.dataComponentProps, children: getNewChildren(children, props, disabledReason, checkboxItemRef) }));
});
export const RadioGroup = forwardRef(function RadioGroup({ children, componentId, analyticsEvents, onValueChange, valueHasNoPii, ...props }, ref) {
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.dropdownMenu', false);
    const radioGroupItemRef = useRef(null);
    useImperativeHandle(ref, () => radioGroupItemRef.current);
    const memoizedAnalyticsEvents = useMemo(() => analyticsEvents ??
        (emitOnView
            ? [
                DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
                DesignSystemEventProviderAnalyticsEventTypes.OnView,
            ]
            : [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange]), [analyticsEvents, emitOnView]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.DropdownMenuRadioGroup,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        valueHasNoPii,
    });
    const { elementRef: radioGroupItemOnViewRef } = useNotifyOnFirstView({
        onView: eventContext.onView,
        value: props.value ?? props.defaultValue,
    });
    const mergedRef = useMergeRefs([radioGroupItemRef, radioGroupItemOnViewRef]);
    const onValueChangeWrapper = useCallback((value) => {
        eventContext.onValueChange(value);
        onValueChange?.(value);
    }, [eventContext, onValueChange]);
    return (_jsx(DropdownMenu.RadioGroup, { ref: mergedRef, onValueChange: onValueChangeWrapper, ...props, ...eventContext.dataComponentProps, children: children }));
});
export const ItemIndicator = forwardRef(function ItemIndicator({ children, ...props }, ref) {
    return (_jsx(DropdownMenu.ItemIndicator, { ref: ref, css: (theme) => ({
            marginLeft: -(CONSTANTS.checkboxIconWidth(theme) + CONSTANTS.checkboxPaddingRight(theme)),
            position: 'absolute',
            fontSize: theme.general.iconFontSize,
        }), ...props, children: children ?? (_jsx(CheckIcon, { css: (theme) => ({
                color: theme.colors.textSecondary,
            }) })) }));
});
export const Arrow = forwardRef(function Arrow({ children, ...props }, ref) {
    const { theme } = useDesignSystemTheme();
    return (_jsx(DropdownMenu.Arrow, { css: {
            fill: theme.colors.backgroundPrimary,
            stroke: theme.colors.borderDecorative,
            strokeDashoffset: -CONSTANTS.arrowBottomLength(),
            strokeDasharray: CONSTANTS.arrowBottomLength() + 2 * CONSTANTS.arrowSide(),
            strokeWidth: CONSTANTS.arrowStrokeWidth(),
            // TODO: This is a temporary fix for the alignment of the Arrow;
            // Radix has changed the implementation for v1.0.0 (uses floating-ui)
            // which has new behaviors for alignment that we don't want. Generally
            // we need to fix the arrow to always be aligned to the left of the menu (with
            // offset equal to border radius)
            position: 'relative',
            top: -1,
        }, ref: ref, width: 12, height: 6, ...props, children: children }));
});
export const RadioItem = forwardRef(function RadioItem({ children, disabledReason, ...props }, ref) {
    const radioItemRef = useRef(null);
    useImperativeHandle(ref, () => radioItemRef.current);
    return (_jsx(DropdownMenu.RadioItem, { ref: radioItemRef, css: (theme) => [dropdownItemStyles, checkboxItemStyles(theme)], ...props, children: getNewChildren(children, props, disabledReason, radioItemRef) }));
});
const SubContext = createContext({ isOpen: false });
const useSubContext = () => React.useContext(SubContext);
export const Sub = ({ children, onOpenChange, ...props }) => {
    const [isOpen, setIsOpen] = React.useState(props.defaultOpen ?? false);
    const handleOpenChange = (isOpen) => {
        onOpenChange?.(isOpen);
        setIsOpen(isOpen);
    };
    return (_jsx(DropdownMenu.Sub, { onOpenChange: handleOpenChange, ...props, children: _jsx(SubContext.Provider, { value: { isOpen }, children: children }) }));
};
// UNWRAPPED RADIX-UI-COMPONENTS
export const Group = DropdownMenu.Group;
// EXTRA COMPONENTS
export const HintColumn = forwardRef(function HintColumn({ children, ...props }, ref) {
    return (_jsx("div", { ref: ref, css: [
            metaTextStyles,
            {
                marginLeft: 'auto',
            },
        ], ...props, children: children }));
});
export const HintRow = forwardRef(function HintRow({ children, ...props }, ref) {
    return (_jsx("div", { ref: ref, css: [
            metaTextStyles,
            {
                minWidth: '100%',
            },
        ], ...props, children: children }));
});
export const IconWrapper = forwardRef(function IconWrapper({ children, ...props }, ref) {
    return (_jsx("div", { ref: ref, css: (theme) => ({
            fontSize: 16,
            color: theme.colors.textSecondary,
            paddingRight: theme.spacing.sm,
        }), ...props, children: children }));
});
// CONSTANTS
const CONSTANTS = {
    itemPaddingVertical(theme) {
        // The number from the mocks is the midpoint between constants
        return 0.5 * theme.spacing.xs + 0.5 * theme.spacing.sm;
    },
    itemPaddingHorizontal(theme) {
        return theme.spacing.sm;
    },
    checkboxIconWidth(theme) {
        return theme.general.iconFontSize;
    },
    checkboxPaddingLeft(theme) {
        return theme.spacing.sm + theme.spacing.xs;
    },
    checkboxPaddingRight(theme) {
        return theme.spacing.sm;
    },
    subMenuIconMargin(theme) {
        // Negative margin so the icons can be larger without increasing the overall item height
        const iconMarginVertical = this.itemPaddingVertical(theme) / 2;
        const iconMarginRight = -this.itemPaddingVertical(theme) + theme.spacing.sm * 1.5;
        return `${-iconMarginVertical}px ${-iconMarginRight}px ${-iconMarginVertical}px auto`;
    },
    subMenuIconSize(theme) {
        return theme.spacing.lg;
    },
    arrowBottomLength() {
        // The built in arrow is a polygon: 0,0 30,0 15,10
        return 30;
    },
    arrowHeight() {
        return 10;
    },
    arrowSide() {
        return 2 * (this.arrowHeight() ** 2 * 2) ** 0.5;
    },
    arrowStrokeWidth() {
        // This is eyeballed b/c relative to the svg viewbox coordinate system
        return 2;
    },
};
export const dropdownContentStyles = (theme, useNewShadows, useNewBorderColors) => ({
    backgroundColor: theme.colors.backgroundPrimary,
    color: theme.colors.textPrimary,
    lineHeight: theme.typography.lineHeightBase,
    border: `1px solid ${useNewBorderColors ? theme.colors.border : theme.colors.borderDecorative}`,
    borderRadius: theme.borders.borderRadiusSm,
    padding: `${theme.spacing.xs}px 0`,
    boxShadow: useNewShadows ? theme.shadows.lg : theme.general.shadowLow,
    userSelect: 'none',
    // Allow for scrolling within the dropdown when viewport is too small
    overflowY: 'auto',
    maxHeight: 'var(--radix-dropdown-menu-content-available-height)',
    ...getDarkModePortalStyles(theme, useNewShadows, useNewBorderColors),
    // Ant Design uses 1000s for their zIndex space; this ensures Radix works with that, but
    // we'll likely need to be sure that all Radix components are using the same zIndex going forward.
    //
    // Additionally, there is an issue where macOS overlay scrollbars in Chrome and Safari (sometimes!)
    // overlap other elements with higher zIndex, because the scrollbars themselves have zIndex 9999,
    // so we have to use a higher value than that: https://github.com/databricks/universe/pull/232825
    zIndex: 10000,
    a: importantify({
        color: theme.colors.textPrimary,
        '&:hover, &:focus': {
            color: theme.colors.textPrimary,
            textDecoration: 'none',
        },
    }),
});
const contentStyles = (theme, useNewShadows, useNewBorderColors) => ({
    ...dropdownContentStyles(theme, useNewShadows, useNewBorderColors),
});
export const dropdownItemStyles = (theme) => ({
    padding: `${CONSTANTS.itemPaddingVertical(theme)}px ${CONSTANTS.itemPaddingHorizontal(theme)}px`,
    display: 'flex',
    flexWrap: 'wrap',
    alignItems: 'center',
    outline: 'unset',
    '&:hover': {
        cursor: 'pointer',
    },
    '&:focus': {
        backgroundColor: theme.colors.actionTertiaryBackgroundHover,
        '&:not(:hover)': {
            outline: `2px auto ${theme.colors.actionDefaultBorderFocus}`,
            outlineOffset: '-1px',
        },
    },
    '&[data-disabled]': {
        pointerEvents: 'none',
        color: `${theme.colors.actionDisabledText} !important`,
    },
});
const dangerItemStyles = (theme) => ({
    color: theme.colors.textValidationDanger,
    '&:hover, &:focus': {
        backgroundColor: theme.colors.actionDangerDefaultBackgroundHover,
    },
});
const checkboxItemStyles = (theme) => ({
    position: 'relative',
    paddingLeft: CONSTANTS.checkboxIconWidth(theme) + CONSTANTS.checkboxPaddingLeft(theme) + CONSTANTS.checkboxPaddingRight(theme),
});
const metaTextStyles = (theme) => ({
    color: theme.colors.textSecondary,
    fontSize: theme.typography.fontSizeSm,
    '[data-disabled] &': {
        color: theme.colors.actionDisabledText,
    },
});
export const dropdownSeparatorStyles = (theme) => ({
    height: 1,
    margin: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
    backgroundColor: theme.colors.borderDecorative,
});
//# sourceMappingURL=DropdownMenu.js.map