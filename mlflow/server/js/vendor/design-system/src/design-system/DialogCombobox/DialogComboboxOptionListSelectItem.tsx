import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react';

import { useDialogComboboxContext } from './hooks/useDialogComboboxContext';
import { useDialogComboboxOptionListContext } from './hooks/useDialogComboboxOptionListContext';
import { dialogComboboxLookAheadKeyDown, getKeyboardNavigationFunctions } from './shared';
import { useDesignSystemTheme } from '../Hooks';
import { CheckIcon, InfoIcon } from '../Icon';
import { LegacyTooltip } from '../LegacyTooltip';
import { useSelectContext } from '../Select/hooks/useSelectContext';
import {
  getComboboxOptionItemWrapperStyles,
  getInfoIconStyles,
  getSelectItemWithHintColumnStyles,
  getHintColumnStyles,
  getComboboxOptionLabelStyles,
} from '../_shared_/Combobox';
import type { HTMLDataAttributes } from '../types';

export interface DialogComboboxOptionListSelectItemProps
  extends HTMLDataAttributes,
    Omit<React.HTMLAttributes<HTMLDivElement>, 'onChange'> {
  value: string;
  checked?: boolean;
  disabled?: boolean;
  disabledReason?: React.ReactNode;
  children?: React.ReactNode;
  onChange?: (value: any, event?: React.MouseEvent<HTMLDivElement> | React.KeyboardEvent<HTMLDivElement>) => void;
  hintColumn?: React.ReactNode;
  hintColumnWidthPercent?: number;
  _TYPE?: string;
  icon?: React.ReactNode;
  /**
   * In certain very custom instances you may wish to hide the check; this is not recommended.
   * If the check is hidden, the user will not be able to tell which item is selected.
   */
  dangerouslyHideCheck?: boolean;
}

const DuboisDialogComboboxOptionListSelectItem = forwardRef<HTMLDivElement, DialogComboboxOptionListSelectItemProps>(
  (
    {
      value,
      checked,
      disabledReason,
      onChange,
      hintColumn,
      hintColumnWidthPercent = 50,
      children,
      _TYPE,
      icon,
      dangerouslyHideCheck,
      ...props
    },
    ref,
  ) => {
    const { theme } = useDesignSystemTheme();
    const {
      stayOpenOnSelection,
      isOpen,
      setIsOpen,
      value: existingValue,
      contentWidth,
      textOverflowMode,
      scrollToSelectedElement,
    } = useDialogComboboxContext();
    const { isInsideDialogComboboxOptionList, lookAhead, setLookAhead } = useDialogComboboxOptionListContext();
    const { isSelect } = useSelectContext();

    if (!isInsideDialogComboboxOptionList) {
      throw new Error('`DialogComboboxOptionListSelectItem` must be used within `DialogComboboxOptionList`');
    }

    const itemRef = useRef<HTMLDivElement>(null);
    const prevCheckedRef = useRef(checked);
    useImperativeHandle(ref, () => itemRef.current as HTMLDivElement);

    useEffect(() => {
      if (scrollToSelectedElement && isOpen) {
        // Check if checked didn't change since the last update, otherwise the popover is still open and we don't need to scroll
        if (checked && prevCheckedRef.current === checked) {
          // Wait for the popover to render and scroll to the selected element's position
          const interval = setInterval(() => {
            if (itemRef.current) {
              itemRef.current?.scrollIntoView?.({
                behavior: 'smooth',
                block: 'center',
              });
              clearInterval(interval);
            }
          }, 50);

          return () => clearInterval(interval);
        }
        prevCheckedRef.current = checked;
      }

      return;
    }, [isOpen, scrollToSelectedElement, checked]);

    const handleSelect = (e: React.MouseEvent<HTMLDivElement> | React.KeyboardEvent<HTMLDivElement>) => {
      if (onChange) {
        if (isSelect) {
          onChange({ value, label: typeof children === 'string' ? children : value }, e);
          if (existingValue?.includes(value)) {
            setIsOpen(false);
          }
          return;
        }
        onChange(value, e);

        // On selecting a previously selected value, manually close the popup, top level logic will not be triggered
        if (!stayOpenOnSelection && existingValue?.includes(value)) {
          setIsOpen(false);
        }
      }
    };

    let content: React.ReactNode = children ?? value;
    if (props.disabled && disabledReason) {
      content = (
        <div css={{ display: 'flex' }}>
          <div>{content}</div>
          <LegacyTooltip title={disabledReason} placement="right">
            <span css={getInfoIconStyles(theme)}>
              <InfoIcon aria-label="Disabled status information" aria-hidden="false" />
            </span>
          </LegacyTooltip>
        </div>
      );
    }

    return (
      <div
        ref={itemRef}
        css={[
          getComboboxOptionItemWrapperStyles(theme),
          {
            '&:focus': {
              background: theme.colors.actionTertiaryBackgroundHover,
              outline: 'none',
            },
          },
        ]}
        {...props}
        onClick={(e: React.MouseEvent<HTMLDivElement> | React.KeyboardEvent<HTMLDivElement>) => {
          if (props.disabled) {
            e.preventDefault();
          } else {
            handleSelect(e);
          }
        }}
        tabIndex={-1}
        {...getKeyboardNavigationFunctions(handleSelect, {
          onKeyDown: props.onKeyDown,
          onMouseEnter: props.onMouseEnter,
          onDefaultKeyDown: (e) => dialogComboboxLookAheadKeyDown(e, setLookAhead, lookAhead),
        })}
        role="option"
        aria-selected={checked}
      >
        {!dangerouslyHideCheck &&
          (checked ? <CheckIcon css={{ paddingTop: 2 }} /> : <div style={{ width: 16, flexShrink: 0 }} />)}
        <label
          css={getComboboxOptionLabelStyles({
            theme,
            dangerouslyHideCheck,
            textOverflowMode,
            contentWidth,
            hasHintColumn: Boolean(hintColumn),
          })}
        >
          {icon && (
            <span
              style={{
                position: 'relative',
                top: 1,
                marginRight: theme.spacing.sm,
                color: theme.colors.textSecondary,
              }}
            >
              {icon}
            </span>
          )}
          {hintColumn ? (
            <span css={getSelectItemWithHintColumnStyles(hintColumnWidthPercent)}>
              {content}
              <span css={getHintColumnStyles(theme, Boolean(props.disabled), textOverflowMode)}>{hintColumn}</span>
            </span>
          ) : (
            content
          )}
        </label>
      </div>
    );
  },
);

DuboisDialogComboboxOptionListSelectItem.defaultProps = {
  _TYPE: 'DialogComboboxOptionListSelectItem',
};

export const DialogComboboxOptionListSelectItem = DuboisDialogComboboxOptionListSelectItem;

export { getComboboxOptionItemWrapperStyles, getComboboxOptionLabelStyles } from '../_shared_/Combobox';
