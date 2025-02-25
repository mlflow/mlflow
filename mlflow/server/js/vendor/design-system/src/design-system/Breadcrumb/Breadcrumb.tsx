import { css } from '@emotion/react';
import type { BreadcrumbProps as AntDBreadcrumbProps } from 'antd';
import { Breadcrumb as AntDBreadcrumb } from 'antd';

import { DesignSystemAntDConfigProvider, getAnimationCss } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks/useDesignSystemTheme';
import { ChevronRightIcon } from '../Icon';
import type { DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
import { addDebugOutlineIfEnabled } from '../utils/debug';

export interface BreadcrumbProps
  extends Pick<AntDBreadcrumbProps, 'itemRender' | 'params' | 'routes' | 'className'>,
    HTMLDataAttributes,
    DangerouslySetAntdProps<AntDBreadcrumbProps> {
  /** Include trailing caret */
  includeTrailingCaret?: boolean;
}

interface BreadcrumbInterface extends React.FC<BreadcrumbProps> {
  Item: typeof AntDBreadcrumb.Item;
  Separator: typeof AntDBreadcrumb.Separator;
}

export const Breadcrumb = /* #__PURE__ */ (() => {
  const Breadcrumb: BreadcrumbInterface = ({ dangerouslySetAntdProps, includeTrailingCaret = true, ...props }) => {
    const { theme, classNamePrefix } = useDesignSystemTheme();

    const separatorClass = `.${classNamePrefix}-breadcrumb-separator`;

    const styles = css({
      // `antd` forces the last anchor to be black, so that it doesn't look like an anchor
      // (even though it is one). This undoes that; if the user wants to make the last
      // text-colored, they can do that by not using an anchor.
      'span:last-child a': {
        color: theme.colors.primary,

        // TODO: Need to pull a global color for anchor hover/focus. Discuss with Ginny.
        ':hover, :focus': {
          color: '#2272B4',
        },
      },

      // TODO: Consider making this global within dubois components
      a: {
        '&:focus-visible': {
          outlineColor: `${theme.colors.actionDefaultBorderFocus} !important`,
          outlineStyle: 'auto !important',
        },
      },

      [separatorClass]: {
        fontSize: theme.general.iconFontSize,
      },

      '& > span': {
        display: 'inline-flex',
        alignItems: 'center',
      },
    });

    return (
      <DesignSystemAntDConfigProvider>
        <AntDBreadcrumb
          {...addDebugOutlineIfEnabled()}
          separator={<ChevronRightIcon />}
          {...props}
          {...dangerouslySetAntdProps}
          css={css(getAnimationCss(theme.options.enableAnimation), styles)}
        >
          {props.children}
          {includeTrailingCaret && props.children && <Breadcrumb.Item> </Breadcrumb.Item>}
        </AntDBreadcrumb>
      </DesignSystemAntDConfigProvider>
    );
  };

  Breadcrumb.Item = AntDBreadcrumb.Item;
  Breadcrumb.Separator = AntDBreadcrumb.Separator;
  return Breadcrumb;
})();
