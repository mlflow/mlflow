import type { SerializedStyles } from '@emotion/react';
import { css } from '@emotion/react';
import classnames from 'classnames';

import type { Theme } from '../../theme';
import { useDesignSystemTheme } from '../Hooks';
import type { InfoPopoverProps } from '../Popover';
import { InfoPopover } from '../Popover';
import type { HTMLDataAttributes } from '../types';
import { addDebugOutlineIfEnabled } from '../utils/debug';

export interface LabelProps extends React.LabelHTMLAttributes<HTMLLabelElement>, HTMLDataAttributes {
  inline?: boolean;
  required?: boolean;
  /**
   * Adds an `InfoPopover` after the label. Please utilize `FormUI.Hint` unless a popover is absolutely necessary.
   * @type React.ReactNode | undefined
   */
  infoPopoverContents?: React.ReactNode;
  infoPopoverProps?: InfoPopoverProps;
}

const getLabelStyles = (theme: Theme, { inline }: { inline?: boolean }): SerializedStyles => {
  const styles = {
    '&&': {
      color: theme.colors.textPrimary,
      fontWeight: theme.typography.typographyBoldFontWeight,
      display: inline ? 'inline' : 'block',
      lineHeight: theme.typography.lineHeightBase,
    },
  };

  return css(styles);
};

const getLabelWrapperStyles = (classNamePrefix: string, theme: Theme): SerializedStyles => {
  const styles = {
    display: 'flex',
    gap: theme.spacing.xs,
    alignItems: 'center',

    [`&& + .${classNamePrefix}-input, && + .${classNamePrefix}-input-affix-wrapper, && + .${classNamePrefix}-select, && + .${classNamePrefix}-selectv2, && + .${classNamePrefix}-dialogcombobox, && + .${classNamePrefix}-checkbox-group, && + .${classNamePrefix}-radio-group, && + .${classNamePrefix}-typeahead-combobox, && + .${classNamePrefix}-datepicker, && + .${classNamePrefix}-rangepicker`]:
      {
        marginTop: theme.spacing.sm,
      },
  };

  return css(styles);
};
export const Label = (props: LabelProps) => {
  const { children, className, inline, required, infoPopoverContents, infoPopoverProps = {}, ...restProps } = props; // Destructure the new prop
  const { classNamePrefix, theme } = useDesignSystemTheme();

  const label = (
    <label
      {...addDebugOutlineIfEnabled()}
      css={[
        getLabelStyles(theme, { inline }),
        ...(!infoPopoverContents ? [getLabelWrapperStyles(classNamePrefix, theme)] : []),
      ]}
      className={classnames(`${classNamePrefix}-label`, className)}
      {...restProps}
    >
      <span css={{ display: 'flex', alignItems: 'center' }}>
        {children}
        {required && <span aria-hidden="true">*</span>}
      </span>
    </label>
  );

  return infoPopoverContents ? (
    <div css={getLabelWrapperStyles(classNamePrefix, theme)}>
      {label}
      <InfoPopover {...infoPopoverProps}>{infoPopoverContents}</InfoPopover>
    </div>
  ) : (
    label
  );
};
