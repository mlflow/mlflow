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

const { Paragraph: AntDParagraph } = AntDTypography;

type AntDTypographyProps = ComponentProps<typeof AntDTypography>;
type AntDParagraphProps = ComponentProps<typeof AntDTypography['Paragraph']>;

export interface TypographyParagraphProps
  extends AntDTypographyProps,
    Pick<AntDParagraphProps, 'ellipsis' | 'disabled' | 'id' | 'title' | 'aria-label'>,
    HTMLDataAttributes,
    DangerouslySetAntdProps<AntDParagraphProps> {
  withoutMargins?: boolean;
  color?: TypographyColor;
}

function getParagraphEmotionStyles(theme: Theme, clsPrefix: string, props: TypographyParagraphProps): SerializedStyles {
  return css(
    {
      '&&': {
        fontSize: theme.typography.fontSizeBase,
        fontWeight: theme.typography.typographyRegularFontWeight,
        lineHeight: theme.typography.lineHeightBase,
        color: getTypographyColor(theme, props.color, theme.colors.textPrimary),
      },
      '& .anticon': {
        verticalAlign: 'text-bottom',
      },
      [`& .${clsPrefix}-btn-link`]: {
        verticalAlign: 'baseline !important',
      },
    },
    props.disabled && { '&&': { color: theme.colors.actionDisabledText } },
    props.withoutMargins && {
      '&&': {
        marginTop: 0,
        marginBottom: 0,
      },
    },
  );
}

export function Paragraph(userProps: TypographyParagraphProps): JSX.Element {
  const { dangerouslySetAntdProps, withoutMargins, color, ...props } = userProps;
  const { theme, classNamePrefix } = useDesignSystemTheme();

  return (
    <DesignSystemAntDConfigProvider>
      <AntDParagraph
        {...addDebugOutlineIfEnabled()}
        {...props}
        className={props.className}
        css={getParagraphEmotionStyles(theme, classNamePrefix, userProps)}
        {...dangerouslySetAntdProps}
      />
    </DesignSystemAntDConfigProvider>
  );
}
