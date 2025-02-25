import { css, type SerializedStyles } from '@emotion/react';
import type { UseMultipleSelectionGetSelectedItemPropsOptions } from 'downshift';
import { forwardRef } from 'react';

import type { Theme } from '../../theme';
import { useDesignSystemTheme } from '../Hooks';
import { CloseIcon } from '../Icon';

export interface TypeaheadComboboxSelectedItemProps<T> {
  label: React.ReactNode;
  item: any;
  getSelectedItemProps: (options: UseMultipleSelectionGetSelectedItemPropsOptions<T>) => any;
  removeSelectedItem: (item: T) => void;
  disabled?: boolean;
}

export const getSelectedItemStyles = (theme: Theme, disabled?: boolean): SerializedStyles => {
  return css({
    backgroundColor: theme.colors.tagDefault,
    borderRadius: theme.general.borderRadiusBase,
    color: theme.colors.textPrimary,
    lineHeight: theme.typography.lineHeightBase,
    fontSize: theme.typography.fontSizeBase,
    marginTop: 2,
    marginBottom: 2,
    marginInlineEnd: theme.spacing.xs,
    paddingRight: 0,
    paddingTop: 0,
    paddingBottom: 0,
    paddingInlineStart: theme.spacing.xs,
    position: 'relative',
    flex: 'none',
    maxWidth: '100%',

    ...(disabled && {
      pointerEvents: 'none',
    }),
  });
};

const getIconContainerStyles = (theme: Theme, disabled?: boolean): SerializedStyles => {
  return css({
    width: 16,
    height: 16,

    ':hover': {
      color: theme.colors.actionTertiaryTextHover,
      backgroundColor: theme.colors.tagHover,
    },

    ...(disabled && {
      pointerEvents: 'none',
      color: theme.colors.actionDisabledText,
    }),
  });
};

const getXIconStyles = (theme: Theme): SerializedStyles => {
  return css({
    fontSize: theme.typography.fontSizeSm,
    verticalAlign: '-1px',
    paddingLeft: theme.spacing.xs / 2,
    paddingRight: theme.spacing.xs / 2,
  });
};

export const TypeaheadComboboxSelectedItem: React.FC<any> = forwardRef<
  HTMLSpanElement,
  TypeaheadComboboxSelectedItemProps<unknown>
>(({ label, item, getSelectedItemProps, removeSelectedItem, disabled, ...restProps }, ref) => {
  const { theme } = useDesignSystemTheme();

  return (
    <span
      {...getSelectedItemProps({ selectedItem: item })}
      css={getSelectedItemStyles(theme, disabled)}
      ref={ref}
      {...restProps}
    >
      <span css={{ marginRight: 2, ...(disabled && { color: theme.colors.actionDisabledText }) }}>{label}</span>
      <span css={getIconContainerStyles(theme, disabled)}>
        <CloseIcon
          aria-hidden="false"
          onClick={(e) => {
            if (!disabled) {
              e.stopPropagation();
              removeSelectedItem(item);
            }
          }}
          css={getXIconStyles(theme)}
          role="button"
          aria-label="Remove selected item"
        />
      </span>
    </span>
  );
});
