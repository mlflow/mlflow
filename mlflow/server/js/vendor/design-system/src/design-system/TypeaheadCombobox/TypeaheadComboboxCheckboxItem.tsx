import type { UseComboboxReturnValue } from 'downshift';
import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react';

import { getMenuItemStyles, type TypeaheadComboboxMenuItemProps } from './TypeaheadComboboxMenuItem';
import { Checkbox } from '../Checkbox';
import { useDesignSystemTheme } from '../Hooks';
import { LegacyInfoTooltip } from '../LegacyTooltip';
import {
  getComboboxOptionItemWrapperStyles,
  HintRow,
  getInfoIconStyles,
  getCheckboxStyles,
} from '../_shared_/Combobox';

export interface TypeaheadComboboxCheckboxItemProps<T> extends TypeaheadComboboxMenuItemProps<T> {
  comboboxState: UseComboboxReturnValue<T>;
  selectedItems: T[];
  _TYPE?: string;
}

export const TypeaheadComboboxCheckboxItem = forwardRef<HTMLLIElement, TypeaheadComboboxCheckboxItemProps<any>>(
  (
    {
      item,
      index,
      comboboxState,
      selectedItems,
      textOverflowMode = 'multiline',
      isDisabled,
      disabledReason,
      hintContent,
      onClick: onClickProp,
      children,
      ...restProps
    },
    ref,
  ) => {
    const { highlightedIndex, getItemProps, isOpen } = comboboxState;
    const isHighlighted = highlightedIndex === index;
    const { theme } = useDesignSystemTheme();
    const isSelected = selectedItems.includes(item);
    const listItemRef = useRef<HTMLLIElement>(null);
    useImperativeHandle(ref, () => listItemRef.current as HTMLLIElement);

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

    const handleClick = (e: React.MouseEvent<HTMLLIElement, MouseEvent>) => {
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

    return (
      <li
        role="option"
        aria-selected={isSelected}
        disabled={isDisabled}
        onClick={handleClick}
        css={[getComboboxOptionItemWrapperStyles(theme), getMenuItemStyles(theme, isHighlighted, isDisabled)]}
        {...downshiftItemProps}
        {...restProps}
      >
        <Checkbox
          componentId="codegen_design-system_src_design-system_typeaheadcombobox_typeaheadcomboboxcheckboxitem.tsx_92"
          disabled={isDisabled}
          isChecked={isSelected}
          css={getCheckboxStyles(theme, textOverflowMode)}
          tabIndex={-1}
          // Needed because Antd handles keyboard inputs as clicks
          onClick={(e) => {
            e.stopPropagation();
          }}
        >
          <label>
            {isDisabled && disabledReason ? (
              <div css={{ display: 'flex' }}>
                <div>{children}</div>
                <div css={getInfoIconStyles(theme)}>
                  <LegacyInfoTooltip title={disabledReason} />
                </div>
              </div>
            ) : (
              children
            )}
            <HintRow disabled={isDisabled}>{hintContent}</HintRow>
          </label>
        </Checkbox>
      </li>
    );
  },
);

TypeaheadComboboxCheckboxItem.defaultProps = {
  _TYPE: 'TypeaheadComboboxCheckboxItem',
};

export default TypeaheadComboboxCheckboxItem;
