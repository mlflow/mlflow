import type { SerializedStyles } from '@emotion/react';
import { css } from '@emotion/react';
import classnames from 'classnames';

import type { Theme } from '../../theme';
import { useDesignSystemTheme } from '../Hooks';
import { addDebugOutlineIfEnabled } from '../utils/debug';

const getHintStyles = (classNamePrefix: string, theme: Theme): SerializedStyles => {
  const styles = {
    display: 'block',
    color: theme.colors.textSecondary,
    lineHeight: theme.typography.lineHeightSm,
    fontSize: theme.typography.fontSizeSm,

    [`&& + .${classNamePrefix}-input, && + .${classNamePrefix}-input-affix-wrapper, && + .${classNamePrefix}-select, && + .${classNamePrefix}-selectv2, && + .${classNamePrefix}-dialogcombobox, && + .${classNamePrefix}-checkbox-group, && + .${classNamePrefix}-radio-group, && + .${classNamePrefix}-typeahead-combobox, && + .${classNamePrefix}-datepicker, && + .${classNamePrefix}-rangepicker`]:
      {
        marginTop: theme.spacing.sm,
      },
  };

  return css(styles);
};

export const Hint = (props: React.HTMLAttributes<HTMLSpanElement>) => {
  const { classNamePrefix, theme } = useDesignSystemTheme();
  const { className, ...restProps } = props;

  return (
    <span
      {...addDebugOutlineIfEnabled()}
      className={classnames(`${classNamePrefix}-hint`, className)}
      css={getHintStyles(classNamePrefix, theme)}
      {...restProps}
    />
  );
};
