import type { TypographyProps as AntDTypographyProps } from 'antd';
import { Typography as AntDTypography } from 'antd';
import type { ReactNode } from 'react';

import { Hint } from './Hint';
import { Link } from './Link';
import { Paragraph } from './Paragraph';
import { Text } from './Text';
import { TextMiddleElide } from './TextMiddleElide';
import { Title } from './Title';
import { Truncate } from './Truncate';
import { DesignSystemAntDConfigProvider } from '../DesignSystemProvider';
import type { DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
import { addDebugOutlineIfEnabled } from '../utils/debug';

interface TypographyProps
  extends Omit<AntDTypographyProps, 'Text' | 'Title' | 'Paragraph' | 'Link'>,
    DangerouslySetAntdProps<AntDTypographyProps>,
    HTMLDataAttributes {
  children?: ReactNode;
  withoutMargins?: boolean;
}

export const Typography = /* #__PURE__ */ (() => {
  function Typography({ dangerouslySetAntdProps, ...props }: TypographyProps): JSX.Element {
    return (
      <DesignSystemAntDConfigProvider>
        <AntDTypography {...addDebugOutlineIfEnabled()} {...props} {...dangerouslySetAntdProps} />
      </DesignSystemAntDConfigProvider>
    );
  }

  Typography.Text = Text;
  Typography.Title = Title;
  Typography.Paragraph = Paragraph;
  Typography.Link = Link;
  Typography.Hint = Hint;
  Typography.Truncate = Truncate;
  Typography.TextMiddleElide = TextMiddleElide;

  return Typography;
})();
