import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react';
import { getMenuItemStyles } from './TypeaheadComboboxMenuItem';
import { Checkbox } from '../Checkbox';
import { useDesignSystemTheme } from '../Hooks';
import { LegacyInfoTooltip } from '../LegacyTooltip';
import { getComboboxOptionItemWrapperStyles, HintRow, getInfoIconStyles, getCheckboxStyles, } from '../_shared_/Combobox';
export const TypeaheadComboboxCheckboxItem = forwardRef(({ item, index, comboboxState, selectedItems, textOverflowMode = 'multiline', isDisabled, disabledReason, hintContent, onClick: onClickProp, children, ...restProps }, ref) => {
    const { highlightedIndex, getItemProps, isOpen } = comboboxState;
    const isHighlighted = highlightedIndex === index;
    const { theme } = useDesignSystemTheme();
    const isSelected = selectedItems.includes(item);
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
        onClick(e);
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
    return (_jsx("li", { role: "option", "aria-selected": isSelected, disabled: isDisabled, onClick: handleClick, css: [getComboboxOptionItemWrapperStyles(theme), getMenuItemStyles(theme, isHighlighted, isDisabled)], ...downshiftItemProps, ...restProps, children: _jsx(Checkbox, { componentId: "codegen_design-system_src_design-system_typeaheadcombobox_typeaheadcomboboxcheckboxitem.tsx_92", disabled: isDisabled, isChecked: isSelected, css: getCheckboxStyles(theme, textOverflowMode), tabIndex: -1, 
            // Needed because Antd handles keyboard inputs as clicks
            onClick: (e) => {
                e.stopPropagation();
            }, children: _jsxs("label", { children: [isDisabled && disabledReason ? (_jsxs("div", { css: { display: 'flex' }, children: [_jsx("div", { children: children }), _jsx("div", { css: getInfoIconStyles(theme), children: _jsx(LegacyInfoTooltip, { title: disabledReason }) })] })) : (children), _jsx(HintRow, { disabled: isDisabled, children: hintContent })] }) }) }));
});
TypeaheadComboboxCheckboxItem.defaultProps = {
    _type: 'TypeaheadComboboxCheckboxItem',
};
export default TypeaheadComboboxCheckboxItem;
//# sourceMappingURL=TypeaheadComboboxCheckboxItem.js.map