import type { CSSObject, SerializedStyles } from '@emotion/react';
import { css } from '@emotion/react';
import React from 'react';
import type { ReactElement } from 'react';

import type { Theme } from '../../theme';
import { useDesignSystemTheme } from '../Hooks';
import { ListIcon } from '../Icon';
import { Typography } from '../Typography';
import type { DangerousGeneralProps, HTMLDataAttributes } from '../types';
import { addDebugOutlineIfEnabled } from '../utils/debug';

const { Title, Paragraph } = Typography;

export interface EmptyProps extends HTMLDataAttributes, DangerousGeneralProps {
  image?: JSX.Element;
  title?: React.ReactNode;
  description: React.ReactNode;
  button?: React.ReactNode;
}

function getEmptyStyles(theme: Theme): SerializedStyles {
  const styles: CSSObject = {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    textAlign: 'center',
    maxWidth: 600,
    wordBreak: 'break-word',

    // TODO: This isn't ideal, but migrating to a safer selector would require a SAFE flag / careful migration.
    '> [role="img"]': {
      // Set size of image to 64px
      fontSize: 64,
      color: theme.colors.actionDisabledText,
      marginBottom: theme.spacing.md,
    },
  };

  return css(styles);
}

function getEmptyTitleStyles(theme: Theme, clsPrefix: string): SerializedStyles {
  const styles: CSSObject = {
    [`&.${clsPrefix}-typography`]: {
      color: theme.colors.textSecondary,
      marginTop: 0,
      marginBottom: 0,
    },
  };

  return css(styles);
}

function getEmptyDescriptionStyles(theme: Theme, clsPrefix: string): SerializedStyles {
  const styles: CSSObject = {
    [`&.${clsPrefix}-typography`]: {
      color: theme.colors.textSecondary,
      marginBottom: theme.spacing.md,
    },
  };

  return css(styles);
}

export const Empty: React.FC<EmptyProps> = (props: EmptyProps): ReactElement => {
  const { theme, classNamePrefix } = useDesignSystemTheme();
  const { title, description, image = <ListIcon />, button, dangerouslyAppendEmotionCSS, ...dataProps } = props;

  return (
    <div {...dataProps} {...addDebugOutlineIfEnabled()} css={{ display: 'flex', justifyContent: 'center' }}>
      <div css={[getEmptyStyles(theme), dangerouslyAppendEmotionCSS]}>
        {image}
        {title && (
          <Title level={3} css={getEmptyTitleStyles(theme, classNamePrefix)}>
            {title}
          </Title>
        )}
        <Paragraph css={getEmptyDescriptionStyles(theme, classNamePrefix)}>{description}</Paragraph>
        {button}
      </div>
    </div>
  );
};
