import { css } from '@emotion/react';
import type { UseComboboxReturnValue } from 'downshift';
import { isEqual } from 'lodash';
import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react';

import type { Theme } from '../../theme';
import { useDesignSystemTheme } from '../Hooks';
import { CheckIcon } from '../Icon';
import { LegacyInfoTooltip } from '../LegacyTooltip';
import { getComboboxOptionItemWrapperStyles, HintRow, getInfoIconStyles } from '../_shared_/Combobox';

export interface TypeaheadComboboxMenuItemProps<T> extends React.HTMLAttributes<HTMLElement> {
  item: T;
  index: number;
  comboboxState: UseComboboxReturnValue<T>;
  textOverflowMode?: 'ellipsis' | 'multiline';
  isDisabled?: boolean;
  disabledReason?: React.ReactNode;
  hintContent?: React.ReactNode;
  className?: string;
  onClick?: (e: React.MouseEvent<HTMLLIElement, MouseEvent>) => void;
  children?: React.ReactNode;
  _TYPE?: string;
}

export const getMenuItemStyles = (theme: Theme, isHighlighted: boolean, disabled?: boolean) => {
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

const getLabelStyles = (theme: Theme, textOverflowMode: 'ellipsis' | 'multiline') => {
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

export const TypeaheadComboboxMenuItem = forwardRef<HTMLLIElement, TypeaheadComboboxMenuItemProps<any>>(
  <T,>(
    {
      item,
      index,
      comboboxState,
      textOverflowMode = 'multiline',
      isDisabled,
      disabledReason,
      hintContent,
      onClick: onClickProp,
      children,
      ...restProps
    }: TypeaheadComboboxMenuItemProps<T>,
    ref: React.Ref<HTMLLIElement>,
  ) => {
    const { selectedItem, highlightedIndex, getItemProps, isOpen } = comboboxState;
    const isSelected = isEqual(selectedItem, item);
    const isHighlighted = highlightedIndex === index;
    const { theme } = useDesignSystemTheme();
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

    return (
      <li
        role="option"
        aria-selected={isSelected}
        aria-disabled={isDisabled}
        onClick={handleClick}
        css={[getComboboxOptionItemWrapperStyles(theme), getMenuItemStyles(theme, isHighlighted, isDisabled)]}
        {...downshiftItemProps}
        {...restProps}
      >
        {isSelected ? <CheckIcon css={{ paddingTop: 2 }} /> : <div style={{ width: 16, flexShrink: 0 }} />}
        <label css={getLabelStyles(theme, textOverflowMode)}>
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
      </li>
    );
  },
) as <T>(props: TypeaheadComboboxMenuItemProps<T> & { ref?: React.Ref<HTMLLIElement> }) => JSX.Element;

(TypeaheadComboboxMenuItem as any).defaultProps = {
  _TYPE: 'TypeaheadComboboxMenuItem',
};

export default TypeaheadComboboxMenuItem;
