import type { SerializedStyles } from '@emotion/react';
import { css } from '@emotion/react';
import { Typography as AntDTypography } from 'antd';
import type { ComponentProps } from 'react';

import type { Theme } from '../../theme';
import { DesignSystemAntDConfigProvider } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks';
import type { DangerouslySetAntdProps, TypographyColor, HTMLDataAttributes } from '../types';
import { addDebugOutlineIfEnabled } from '../utils/debug';

const { Text: AntDText } = AntDTypography;

type AntDTypographyProps = ComponentProps<typeof AntDTypography>;
type AntDTextProps = ComponentProps<typeof AntDTypography['Text']>;

export interface TypographyHintProps
  extends AntDTypographyProps,
    Pick<AntDTextProps, 'ellipsis' | 'id' | 'title' | 'aria-label'>,
    HTMLDataAttributes,
    DangerouslySetAntdProps<Omit<AntDTextProps, 'hint'>> {
  bold?: boolean;
  size?: 'sm' | 'md' | 'lg' | 'xl' | 'xxl';
  withoutMargins?: boolean;
  color?: TypographyColor;
}

function getTextEmotionStyles(theme: Theme, props: TypographyHintProps): SerializedStyles {
  return css({
    '&&': {
      display: 'block',
      fontSize: theme.typography.fontSizeSm,
      lineHeight: theme.typography.lineHeightSm,
      color: theme.colors.textSecondary,

      ...(props.withoutMargins && {
        '&&': {
          marginTop: 0,
          marginBottom: 0,
        },
      }),
    },
  });
}

export function Hint(userProps: TypographyHintProps): JSX.Element {
  const { dangerouslySetAntdProps, bold, withoutMargins, color, ...props } = userProps;
  const { theme } = useDesignSystemTheme();

  return (
    <DesignSystemAntDConfigProvider>
      <AntDText
        {...addDebugOutlineIfEnabled()}
        {...props}
        css={getTextEmotionStyles(theme, userProps)}
        {...dangerouslySetAntdProps}
      />
    </DesignSystemAntDConfigProvider>
  );
}
