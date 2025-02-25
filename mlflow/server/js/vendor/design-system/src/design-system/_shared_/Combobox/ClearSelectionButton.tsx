import { css } from '@emotion/react';
import type { SerializedStyles } from '@emotion/react';

import type { Theme } from '../../../theme';
import { useDesignSystemTheme } from '../../Hooks';
import { XCircleFillIcon, type IconProps } from '../../Icon';

export interface ClearSelectionButtonProps extends IconProps {}

const getButtonStyles = (theme: Theme): SerializedStyles => {
  return css({
    color: theme.colors.textPlaceholder,
    fontSize: theme.typography.fontSizeSm,
    marginLeft: theme.spacing.xs,

    ':hover': {
      color: theme.colors.actionTertiaryTextHover,
    },
  });
};

export const ClearSelectionButton = ({ ...restProps }: ClearSelectionButtonProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <XCircleFillIcon
      aria-hidden="false"
      css={getButtonStyles(theme)}
      role="button"
      aria-label="Clear selection"
      {...restProps}
    />
  );
};
