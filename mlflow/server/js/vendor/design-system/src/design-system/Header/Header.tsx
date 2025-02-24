import type { SerializedStyles } from '@emotion/react';
import { css } from '@emotion/react';
import React from 'react';

import type { Theme } from '../../theme';
import { useDesignSystemTheme } from '../Hooks';
import { Space } from '../Space';
import type { TypographyTitleProps } from '../Typography/Title';
import { Title } from '../Typography/Title';
import type { DangerousGeneralProps, HTMLDataAttributes } from '../types';
import { importantify } from '../utils/css-utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';

export interface HeaderProps extends HTMLDataAttributes, DangerousGeneralProps {
  /** The title for this page */
  title: React.ReactNode;
  /** Inline elements to be appended to the end of the title, such as a `Tag` */
  titleAddOns?: React.ReactNode | React.ReactNode[];
  /** A single `<Breadcrumb />` component */
  breadcrumbs?: React.ReactNode;
  /** An array of Dubois `<Button />` components */
  buttons?: React.ReactNode | React.ReactNode[];
  /** HTML title element level. This only controls the element rendered, title will look like a h2 */
  titleElementLevel?: TypographyTitleProps['elementLevel'];
}

const getHeaderStyles = (clsPrefix: string, theme: Theme): SerializedStyles => {
  const breadcrumbClass = `.${clsPrefix}-breadcrumb`;

  const styles = {
    [breadcrumbClass]: {
      lineHeight: theme.typography.lineHeightBase,
    },
  };

  return css(importantify(styles));
};

export const Header: React.FC<HeaderProps> = ({
  breadcrumbs,
  title,
  titleAddOns,
  dangerouslyAppendEmotionCSS,
  buttons,
  children,
  titleElementLevel,
  ...rest
}) => {
  const { classNamePrefix: clsPrefix, theme } = useDesignSystemTheme();
  const buttonsArray: React.ReactNode[] = Array.isArray(buttons) ? buttons : buttons ? [buttons] : [];

  // TODO: Move to getHeaderStyles for consistency, followup ticket: https://databricks.atlassian.net/browse/FEINF-1222
  const styles = {
    titleWrapper: css({
      display: 'flex',
      alignItems: 'flex-start',
      justifyContent: 'space-between',
      flexWrap: 'wrap',
      rowGap: theme.spacing.sm,
      // Buttons have 32px height while Title level 2 elements used by this component have a height of 28px
      // These paddings enforce height to be the same without buttons too
      ...(buttonsArray.length === 0 && {
        paddingTop: breadcrumbs ? 0 : theme.spacing.xs / 2,
        paddingBottom: theme.spacing.xs / 2,
      }),
    }),

    breadcrumbWrapper: css({
      lineHeight: theme.typography.lineHeightBase,
      marginBottom: theme.spacing.xs,
    }),

    title: css({
      marginTop: 0,
      marginBottom: '0 !important',
      alignSelf: 'stretch',
    }),

    // TODO: Look into a more emotion-idomatic way of doing this.
    titleIfOtherElementsPresent: css({
      marginTop: 2,
    }),

    buttonContainer: css({
      marginLeft: 8,
    }),

    titleAddOnsWrapper: css({
      display: 'inline-flex',
      verticalAlign: 'middle',
      alignItems: 'center',
      flexWrap: 'wrap',
      marginLeft: theme.spacing.sm,
      gap: theme.spacing.xs,
    }),
  };

  return (
    <div
      {...addDebugOutlineIfEnabled()}
      css={[getHeaderStyles(clsPrefix, theme), dangerouslyAppendEmotionCSS]}
      {...rest}
    >
      {breadcrumbs && <div css={styles.breadcrumbWrapper}>{breadcrumbs}</div>}
      <div css={styles.titleWrapper}>
        <Title
          level={2}
          elementLevel={titleElementLevel}
          css={[styles.title, (buttons || breadcrumbs) && styles.titleIfOtherElementsPresent]}
        >
          {title}
          {titleAddOns && <span css={styles.titleAddOnsWrapper}>{titleAddOns}</span>}
        </Title>
        {buttons && (
          <div css={styles.buttonContainer}>
            {/* TODO: I'm using the deprecated `Space` component, but
            this is actually a decent use-case. I'll investigate as a follow-up */}
            <Space dangerouslySetAntdProps={{ wrap: true }} size={8}>
              {buttonsArray.filter(Boolean).map((button, i) => {
                const defaultKey = `dubois-header-button-${i}`;

                return React.isValidElement(button) ? (
                  React.cloneElement(button, {
                    key: button.key || defaultKey,
                  })
                ) : (
                  <React.Fragment key={defaultKey}>{button}</React.Fragment>
                );
              })}
            </Space>
          </div>
        )}
      </div>
    </div>
  );
};
