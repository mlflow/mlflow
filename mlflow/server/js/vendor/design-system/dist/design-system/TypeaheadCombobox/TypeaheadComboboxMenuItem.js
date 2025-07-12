import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { isEqual } from 'lodash';
import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react';
import { useDesignSystemTheme } from '../Hooks';
import { CheckIcon } from '../Icon';
import { LegacyInfoTooltip } from '../LegacyTooltip';
import { getComboboxOptionItemWrapperStyles, HintRow, getInfoIconStyles } from '../_shared_/Combobox';
export const getMenuItemStyles = (theme, isHighlighted, disabled) => {
    return css({
        ...(disabled && {
            pointerEvents: 'none',
            color: theme.colors.actionDisabledText,
        }),
        ...(isHighlighted && {
            background: theme.colors.actionTertiaryBackgroundHover,
        }),
    });
};
const getLabelStyles = (theme, textOverflowMode) => {
    return css({
        marginLeft: theme.spacing.sm,
        fontSize: theme.typography.fontSizeBase,
        fontStyle: 'normal',
        fontWeight: 400,
        cursor: 'pointer',
        overflow: 'hidden',
        wordBreak: 'break-word',
        ...(textOverflowMode === 'ellipsis' && {
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
        }),
    });
};
export const TypeaheadComboboxMenuItem = forwardRef(({ item, index, comboboxState, textOverflowMode = 'multiline', isDisabled, disabledReason, hintContent, onClick: onClickProp, children, ...restProps }, ref) => {
    const { selectedItem, highlightedIndex, getItemProps, isOpen } = comboboxState;
    const isSelected = isEqual(selectedItem, item);
    const isHighlighted = highlightedIndex === index;
    const { theme } = useDesignSystemTheme();
    const listItemRef = useRef(null);
    useImperativeHandle(ref, () => listItemRef.current);
    const { onClick, ...downshiftItemProps } = getItemProps({
        item,
        index,
        disabled: isDisabled,
        onMouseUp: (e) => {
            e.stopPropagation();
            restProps.onMouseUp?.(e);
        },
        ref: listItemRef,
    });
    const handleClick = (e) => {
        onClickProp?.(e);
        onClick?.(e);
    };
    // Scroll to the highlighted item if it is not in the viewport
    useEffect(() => {
        if (isOpen && highlightedIndex === index && listItemRef.current) {
            const parentContainer = listItemRef.current.closest('ul');
            if (!parentContainer) {
                return;
            }
            const parentTop = parentContainer.scrollTop;
            const parentBottom = parentContainer.scrollTop + parentContainer.clientHeight;
            const itemTop = listItemRef.current.offsetTop;
            const itemBottom = listItemRef.current.offsetTop + listItemRef.current.clientHeight;
            // Check if item is visible in the viewport before scrolling
            if (itemTop < parentTop || itemBottom > parentBottom) {
                listItemRef.current?.scrollIntoView({ block: 'nearest' });
            }
        }
    }, [highlightedIndex, index, isOpen, listItemRef]);
    return (_jsxs("li", { role: "option", "aria-selected": isSelected, "aria-disabled": isDisabled, onClick: handleClick, css: [getComboboxOptionItemWrapperStyles(theme), getMenuItemStyles(theme, isHighlighted, isDisabled)], ...downshiftItemProps, ...restProps, children: [isSelected ? _jsx(CheckIcon, { css: { paddingTop: 2 } }) : _jsx("div", { style: { width: 16, flexShrink: 0 } }), _jsxs("label", { css: getLabelStyles(theme, textOverflowMode), children: [isDisabled && disabledReason ? (_jsxs("div", { css: { display: 'flex' }, children: [_jsx("div", { children: children }), _jsx("div", { css: getInfoIconStyles(theme), children: _jsx(LegacyInfoTooltip, { title: disabledReason }) })] })) : (children), _jsx(HintRow, { disabled: isDisabled, children: hintContent })] })] }));
});
TypeaheadComboboxMenuItem.defaultProps = {
    _type: 'TypeaheadComboboxMenuItem',
};
export default TypeaheadComboboxMenuItem;
//# sourceMappingURL=TypeaheadComboboxMenuItem.js.map