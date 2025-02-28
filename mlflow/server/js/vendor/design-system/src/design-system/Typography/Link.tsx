import type { Interpolation, SerializedStyles } from '@emotion/react';
import { css } from '@emotion/react';
import { Typography as AntDTypography } from 'antd';
import type { ComponentProps } from 'react';
import { forwardRef, useCallback, useMemo } from 'react';

import type { Theme } from '../../theme';
import {
  DesignSystemEventProviderAnalyticsEventTypes,
  DesignSystemEventProviderComponentTypes,
  useDesignSystemEventComponentCallbacks,
} from '../DesignSystemEventProvider/DesignSystemEventProvider';
import { DesignSystemAntDConfigProvider } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks';
import { NewWindowIcon } from '../Icon';
import type { AnalyticsEventProps, DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
import { addDebugOutlineIfEnabled } from '../utils/debug';

type AntDLinkProps = ComponentProps<typeof AntDTypography['Link']>;

export interface TypographyLinkProps
  extends AntDLinkProps,
    DangerouslySetAntdProps<AntDLinkProps>,
    HTMLDataAttributes,
    AnalyticsEventProps<DesignSystemEventProviderAnalyticsEventTypes.OnClick> {
  /**
   * Configures a link to be opened in a new tab by setting `target` to `'_blank'`
   * and `rel` to `'noopener noreferrer'`, which is necessary for security, and
   * rendering an "external link" icon next to the link when `true`.
   */
  openInNewTab?: boolean;
}

const getLinkStyles = (theme: Theme, clsPrefix: string): SerializedStyles => {
  const classTypography = `.${clsPrefix}-typography`;

  const styles: Interpolation<Theme> = {
    [`&${classTypography}, &${classTypography}:focus`]: {
      color: theme.colors.actionTertiaryTextDefault,
    },

    [`&${classTypography}:hover, &${classTypography}:hover .anticon`]: {
      color: theme.colors.actionTertiaryTextHover,
      textDecoration: 'underline',
    },

    [`&${classTypography}:active, &${classTypography}:active .anticon`]: {
      color: theme.colors.actionTertiaryTextPress,
      textDecoration: 'underline',
    },

    [`&${classTypography}:focus-visible`]: {
      textDecoration: 'underline',
    },

    '.anticon': {
      fontSize: 12,
      verticalAlign: 'baseline',
    },

    // manually update color for link within a LegacyTooltip since tooltip always has an inverted background color for light/dark mode
    // this is required for accessibility compliance
    [`.${clsPrefix}-tooltip-inner a&${classTypography}`]: {
      [`&, :focus`]: {
        color: theme.colors.blue500,
        '.anticon': { color: theme.colors.blue500 },
      },
      ':active': {
        color: theme.colors.blue500,
        '.anticon': { color: theme.colors.blue500 },
      },
      ':hover': {
        color: theme.colors.blue400,
        '.anticon': { color: theme.colors.blue400 },
      },
    },
  };

  return css(styles);
};

const getEllipsisNewTabLinkStyles = (): SerializedStyles => {
  const styles: Interpolation<Theme> = {
    paddingRight: 'calc(2px + 1em)', // 1em for icon
    position: 'relative',
  };

  return css(styles);
};

const getIconStyles = (theme: Theme): SerializedStyles => {
  const styles: Interpolation<Theme> = {
    marginLeft: 4,
    color: theme.colors.actionTertiaryTextDefault,
    position: 'relative',
    top: '1px',
  };

  return css(styles);
};

const getEllipsisIconStyles = (useNewIcons?: boolean): SerializedStyles => {
  const styles: Interpolation<Theme> = {
    position: 'absolute',
    right: 0,
    bottom: 0,
    top: 0,
    display: 'flex',
    alignItems: 'center',
    ...(useNewIcons && {
      fontSize: 12,
    }),
  };

  return css(styles);
};

export const Link = forwardRef<HTMLAnchorElement, TypographyLinkProps>(function Link(
  {
    dangerouslySetAntdProps,
    componentId,
    analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnClick],
    onClick,
    ...props
  },
  ref,
) {
  const { children, openInNewTab, ...restProps } = props;
  const { theme, classNamePrefix } = useDesignSystemTheme();
  const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.TypographyLink,
    componentId,
    analyticsEvents: memoizedAnalyticsEvents,
    shouldStartInteraction: false,
  });

  const onClickHandler: React.MouseEventHandler<HTMLAnchorElement> = useCallback(
    (e) => {
      eventContext.onClick(e);
      onClick?.(e);
    },
    [eventContext, onClick],
  );

  const newTabProps = {
    rel: 'noopener noreferrer',
    target: '_blank',
  };

  const linkProps = openInNewTab ? { ...restProps, ...newTabProps } : { ...restProps };

  const linkStyles =
    props.ellipsis && openInNewTab
      ? [getLinkStyles(theme, classNamePrefix), getEllipsisNewTabLinkStyles()]
      : getLinkStyles(theme, classNamePrefix);
  const iconStyles = props.ellipsis ? [getIconStyles(theme), getEllipsisIconStyles()] : getIconStyles(theme);

  return (
    <DesignSystemAntDConfigProvider>
      <AntDTypography.Link
        {...addDebugOutlineIfEnabled()}
        aria-disabled={linkProps.disabled}
        css={linkStyles}
        ref={ref}
        onClick={onClickHandler}
        {...linkProps}
        {...dangerouslySetAntdProps}
        {...eventContext.dataComponentProps}
      >
        {children}
        {openInNewTab ? <NewWindowIcon css={iconStyles} {...newTabProps} /> : null}
      </AntDTypography.Link>
    </DesignSystemAntDConfigProvider>
  );
});
