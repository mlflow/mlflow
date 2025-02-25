import type { SerializedStyles } from '@emotion/react';
import { css } from '@emotion/react';
import { Typography as AntDTypography } from 'antd';
import type { ComponentProps } from 'react';

import type { Theme } from '../../theme';
import { DesignSystemAntDConfigProvider } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks';
import type { DangerouslySetAntdProps, TypographyColor, HTMLDataAttributes } from '../types';
import { getTypographyColor } from '../utils/css-utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';

const { Title: AntDTitle } = AntDTypography;

type AntDTypographyProps = ComponentProps<typeof AntDTypography>;
type AntDTitleProps = ComponentProps<typeof AntDTypography['Title']>;

export interface TypographyTitleProps
  extends AntDTypographyProps,
    Pick<AntDTitleProps, 'level' | 'ellipsis' | 'id' | 'title' | 'aria-label'>,
    HTMLDataAttributes,
    DangerouslySetAntdProps<AntDTitleProps> {
  withoutMargins?: boolean;
  color?: TypographyColor;
  /** Only controls the HTML element rendered, styles are controlled by `level` prop */
  elementLevel?: AntDTitleProps['level'];
}

function getLevelStyles(theme: Theme, props: TypographyTitleProps): SerializedStyles {
  switch (props.level) {
    case 1:
      return css({
        '&&': {
          fontSize: theme.typography.fontSizeXxl,
          lineHeight: theme.typography.lineHeightXxl,
          fontWeight: theme.typography.typographyBoldFontWeight,
        },
        '& > .anticon': {
          lineHeight: theme.typography.lineHeightXxl,
        },
      });
    case 2:
      return css({
        '&&': {
          fontSize: theme.typography.fontSizeXl,
          lineHeight: theme.typography.lineHeightXl,
          fontWeight: theme.typography.typographyBoldFontWeight,
        },
        '& > .anticon': {
          lineHeight: theme.typography.lineHeightXl,
        },
      });
    case 3:
      return css({
        '&&': {
          fontSize: theme.typography.fontSizeLg,
          lineHeight: theme.typography.lineHeightLg,
          fontWeight: theme.typography.typographyBoldFontWeight,
        },
        '& > .anticon': {
          lineHeight: theme.typography.lineHeightLg,
        },
      });
    case 4:
    default:
      return css({
        '&&': {
          fontSize: theme.typography.fontSizeMd,
          lineHeight: theme.typography.lineHeightMd,
          fontWeight: theme.typography.typographyBoldFontWeight,
        },
        '& > .anticon': {
          lineHeight: theme.typography.lineHeightMd,
        },
      });
  }
}

function getTitleEmotionStyles(theme: Theme, props: TypographyTitleProps): SerializedStyles {
  return css(
    getLevelStyles(theme, props),
    {
      '&&': {
        color: getTypographyColor(theme, props.color, theme.colors.textPrimary),
      },
      '& > .anticon': {
        verticalAlign: 'middle',
      },
    },
    props.withoutMargins && {
      '&&': {
        marginTop: '0 !important', // override general styling
        marginBottom: '0 !important', // override general styling
      },
    },
  );
}
export function Title(userProps: TypographyTitleProps): JSX.Element {
  const { dangerouslySetAntdProps, withoutMargins, color, elementLevel, ...props } = userProps;
  const { theme } = useDesignSystemTheme();

  return (
    <DesignSystemAntDConfigProvider>
      <AntDTitle
        {...addDebugOutlineIfEnabled()}
        {...props}
        level={elementLevel ?? props.level}
        className={props.className}
        css={getTitleEmotionStyles(theme, userProps)}
        {...dangerouslySetAntdProps}
      />
    </DesignSystemAntDConfigProvider>
  );
}
