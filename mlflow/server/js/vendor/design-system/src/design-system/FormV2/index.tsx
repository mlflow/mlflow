import type { SerializedStyles } from '@emotion/react';
import { css } from '@emotion/react';

import type { Theme } from '../../theme';
import { FormMessage } from '../FormMessage/FormMessage';
import { Hint } from '../Hint/Hint';
import { useDesignSystemTheme } from '../Hooks';
import { Label } from '../Label/Label';

export * from './RHFAdapters';

export interface HorizontalFormSectionProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  labelColWidth?: number | string;
  inputColWidth?: number | string;
}

const getHorizontalInputStyles = (
  theme: Theme,
  labelColWidth: number | string,
  inputColWidth: number | string,
): SerializedStyles => {
  return css({
    display: 'flex',
    gap: theme.spacing.sm,

    '& > input, & > textarea, & > select': {
      marginTop: '0 !important',
    },

    '& > div:nth-of-type(1)': {
      width: labelColWidth,
    },

    '& > div:nth-of-type(2)': {
      width: inputColWidth,
    },
  });
};

const HorizontalFormRow = ({
  children,
  labelColWidth = '33%',
  inputColWidth = '66%',
  ...restProps
}: HorizontalFormSectionProps) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div css={getHorizontalInputStyles(theme, labelColWidth, inputColWidth)} {...restProps}>
      {children}
    </div>
  );
};

export const FormUI = {
  Message: FormMessage,
  Label: Label,
  Hint: Hint,
  HorizontalFormRow,
};
