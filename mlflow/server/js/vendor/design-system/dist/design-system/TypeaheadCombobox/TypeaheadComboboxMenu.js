import { jsx as _jsx, Fragment as _Fragment, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { computePosition, flip, size } from '@floating-ui/dom';
import { useMergeRefs } from '@floating-ui/react';
import React, { Children, Fragment, forwardRef, useCallback, useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { useTypeaheadComboboxContext } from './hooks';
import { useDesignSystemTheme } from '../Hooks';
import { useDesignSystemContext } from '../Hooks/useDesignSystemContext';
import { EmptyResults, LoadingSpinner, getComboboxContentWrapperStyles } from '../_shared_/Combobox';
import { useDesignSystemSafexFlags } from '../utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
const getTypeaheadComboboxMenuStyles = () => {
    return css({
        padding: 0,
        margin: 0,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'flex-start',
        position: 'absolute',
    });
};
export const TypeaheadComboboxMenu = forwardRef(({ comboboxState, loading, emptyText, width, minWidth = 240, maxWidth, minHeight, maxHeight, listWrapperHeight, virtualizerRef, children, matchTriggerWidth, ...restProps }, ref) => {
    const { getMenuProps, isOpen } = comboboxState;
    const { ref: downshiftRef, ...downshiftProps } = getMenuProps({}, { suppressRefError: true });
    const { useNewShadows, useNewBorderColors } = useDesignSystemSafexFlags();
    const [viewPortMaxHeight, setViewPortMaxHeight] = useState(undefined);
    const { floatingUiRefs, floatingStyles, isInsideTypeaheadCombobox, inputWidth } = useTypeaheadComboboxContext();
    if (!isInsideTypeaheadCombobox) {
        throw new Error('`TypeaheadComboboxMenu` must be used within `TypeaheadCombobox`');
    }
    const mergedRef = useMergeRefs([ref, downshiftRef, floatingUiRefs?.setFloating, virtualizerRef]);
    const { theme } = useDesignSystemTheme();
    const { getPopupContainer } = useDesignSystemContext();
    const recalculateMaxHeight = useCallback(() => {
        if (isOpen &&
            floatingUiRefs?.floating &&
            floatingUiRefs.reference.current &&
            floatingUiRefs?.reference &&
            floatingUiRefs.floating.current) {
            computePosition(floatingUiRefs.reference.current, floatingUiRefs.floating.current, {
                middleware: [
                    flip(),
                    size({
                        padding: theme.spacing.sm,
                        apply({ availableHeight }) {
                            setViewPortMaxHeight(availableHeight);
                        },
                    }),
                ],
            });
        }
    }, [isOpen, floatingUiRefs, theme.spacing.sm]);
    useEffect(() => {
        if (!isOpen || maxHeight) {
            return;
        }
        recalculateMaxHeight();
        window.addEventListener('scroll', recalculateMaxHeight);
        return () => {
            window.removeEventListener('scroll', recalculateMaxHeight);
        };
    }, [isOpen, maxHeight, recalculateMaxHeight]);
    if (!isOpen)
        return null;
    const hasFragmentWrapper = children && !Array.isArray(children) && children.type === Fragment;
    const filterableChildren = hasFragmentWrapper ? children.props.children : children;
    const hasResults = filterableChildren &&
        Children.toArray(filterableChildren).some((child) => {
            if (React.isValidElement(child)) {
                const childType = child.props['__EMOTION_TYPE_PLEASE_DO_NOT_USE__']?.defaultProps._type ?? child.props._type;
                return ['TypeaheadComboboxMenuItem', 'TypeaheadComboboxCheckboxItem'].includes(childType);
            }
            return false;
        });
    const [menuItemChildren, footer] = Children.toArray(filterableChildren).reduce((acc, child) => {
        const isFooter = React.isValidElement(child) && child.props._type === 'TypeaheadComboboxFooter';
        if (isFooter) {
            acc[1].push(child);
        }
        else {
            acc[0].push(child);
        }
        return acc;
    }, [[], []]);
    return createPortal(_jsx("ul", { ...addDebugOutlineIfEnabled(), "aria-busy": loading, ...downshiftProps, ref: mergedRef, css: [
            getComboboxContentWrapperStyles(theme, {
                maxHeight: maxHeight ?? viewPortMaxHeight,
                maxWidth,
                minHeight,
                minWidth,
                width,
                useNewShadows,
                useNewBorderColors,
            }),
            getTypeaheadComboboxMenuStyles(),
            matchTriggerWidth && inputWidth && { width: inputWidth },
        ], style: { ...floatingStyles }, ...restProps, children: loading ? (_jsx(LoadingSpinner, { "aria-label": "Loading", alt: "Loading spinner" })) : hasResults ? (_jsxs(_Fragment, { children: [_jsx("div", { style: {
                        position: 'relative',
                        width: '100%',
                        ...(listWrapperHeight && { height: listWrapperHeight, flexShrink: 0 }),
                    }, children: menuItemChildren }), footer] })) : (_jsxs(_Fragment, { children: [_jsx(EmptyResults, { emptyText: emptyText }), footer] })) }), getPopupContainer ? getPopupContainer() : document.body);
});
//# sourceMappingURL=TypeaheadComboboxMenu.js.map