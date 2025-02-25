import { forwardRef } from 'react';

import { useDialogComboboxContext } from './hooks/useDialogComboboxContext';
import { useDialogComboboxOptionListContext } from './hooks/useDialogComboboxOptionListContext';
import {
  getKeyboardNavigationFunctions,
  dialogComboboxLookAheadKeyDown,
  getDialogComboboxOptionLabelWidth,
} from './shared';
import { Checkbox } from '../Checkbox';
import { useDesignSystemTheme } from '../Hooks';
import { InfoIcon } from '../Icon';
import { LegacyTooltip } from '../LegacyTooltip';
import { getComboboxOptionItemWrapperStyles, getInfoIconStyles, getCheckboxStyles } from '../_shared_/Combobox';
import type { HTMLDataAttributes } from '../types';

export interface DialogComboboxOptionListCheckboxItemProps
  extends HTMLDataAttributes,
    Omit<React.HTMLAttributes<HTMLDivElement>, 'onChange'> {
  value: string;
  checked?: boolean;
  disabled?: boolean;
  disabledReason?: React.ReactNode;
  indeterminate?: boolean;
  children?: React.ReactNode;
  onChange?: (value: any, event?: React.MouseEvent<HTMLDivElement> | React.KeyboardEvent<HTMLDivElement>) => void;
  _TYPE?: string;
}

const DuboisDialogComboboxOptionListCheckboxItem = forwardRef<
  HTMLDivElement,
  DialogComboboxOptionListCheckboxItemProps
>(({ value, checked, indeterminate, onChange, children, disabledReason, _TYPE, ...props }, ref) => {
  const { theme } = useDesignSystemTheme();
  const { textOverflowMode, contentWidth } = useDialogComboboxContext();
  const { isInsideDialogComboboxOptionList, setLookAhead, lookAhead } = useDialogComboboxOptionListContext();

  if (!isInsideDialogComboboxOptionList) {
    throw new Error('`DialogComboboxOptionListCheckboxItem` must be used within `DialogComboboxOptionList`');
  }

  const handleSelect = (e: React.MouseEvent<HTMLDivElement> | React.KeyboardEvent<HTMLDivElement>) => {
    if (onChange) {
      onChange(value, e);
    }
  };

  let content: React.ReactNode = children ?? value;
  if (props.disabled && disabledReason) {
    content = (
      <div css={{ display: 'flex' }}>
        <div>{content}</div>
        <div>
          <LegacyTooltip title={disabledReason} placement="right">
            <span css={getInfoIconStyles(theme)}>
              <InfoIcon aria-label="Disabled status information" aria-hidden="false" />
            </span>
          </LegacyTooltip>
        </div>
      </div>
    );
  }

  return (
    <div
      ref={ref}
      role="option"
      // Using aria-selected instead of aria-checked because the parent listbox
      aria-selected={indeterminate ? false : checked}
      css={[getComboboxOptionItemWrapperStyles(theme)]}
      {...props}
      onClick={(e) => {
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
    >
      <Checkbox
        componentId="codegen_design-system_src_design-system_dialogcombobox_dialogcomboboxoptionlistcheckboxitem.tsx_86"
        disabled={props.disabled}
        isChecked={indeterminate ? null : checked}
        css={[
          getCheckboxStyles(theme, textOverflowMode),
          contentWidth
            ? {
                '& > span:last-of-type': {
                  width: getDialogComboboxOptionLabelWidth(theme, contentWidth),
                },
              }
            : {},
        ]}
        tabIndex={-1}
        // Needed because Antd handles keyboard inputs as clicks
        onClick={(e) => {
          e.stopPropagation();
          handleSelect(e);
        }}
      >
        {/* `Checkbox` styles assume there's only one child node in `children`, so we need to wrap `content`. */}
        <div css={{ maxWidth: '100%' }}>{content}</div>
      </Checkbox>
    </div>
  );
});

DuboisDialogComboboxOptionListCheckboxItem.defaultProps = {
  _TYPE: 'DialogComboboxOptionListCheckboxItem',
};

export const DialogComboboxOptionListCheckboxItem = DuboisDialogComboboxOptionListCheckboxItem;
