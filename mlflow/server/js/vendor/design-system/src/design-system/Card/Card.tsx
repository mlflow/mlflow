import type { SerializedStyles } from '@emotion/react';
import { css } from '@emotion/react';
import { useMergeRefs } from '@floating-ui/react';
import { isUndefined } from 'lodash';
import type { HTMLAttributes, HTMLProps, PropsWithChildren } from 'react';
import React, { useCallback, useMemo } from 'react';

import type { Theme } from '../../theme';
import {
  DesignSystemEventProviderAnalyticsEventTypes,
  DesignSystemEventProviderComponentTypes,
  useDesignSystemEventComponentCallbacks,
} from '../DesignSystemEventProvider/DesignSystemEventProvider';
import { getAnimationCss } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks';
import { ParagraphSkeleton, TitleSkeleton } from '../Skeleton';
import type { AnalyticsEventPropsWithStartInteraction, DangerousGeneralProps, HTMLDataAttributes } from '../types';
import { useDesignSystemSafexFlags, useNotifyOnFirstView } from '../utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';

export interface CardProps
  extends DangerousGeneralProps,
    HTMLDataAttributes,
    HTMLAttributes<HTMLDivElement>,
    AnalyticsEventPropsWithStartInteraction<
      DesignSystemEventProviderAnalyticsEventTypes.OnClick | DesignSystemEventProviderAnalyticsEventTypes.OnView
    > {
  /**
   * disable the default hover style effects (border, box-shadow)
   *
   * @default false
   */
  disableHover?: boolean;

  /**
   * Show the card in loading state using a Skeleton
   * Without a given width we will default to 300px
   *
   * @default false
   */
  loading?: boolean;

  /**
   * Custom loading content using Dubois Skeleton components
   *
   * @default undefined
   */
  customLoadingContent?: React.ReactNode;

  /**
   *
   * @default undefined
   */
  onClick?: (e?: React.MouseEvent | React.KeyboardEvent) => void;

  /**
   * Width of the card, used when displaying a list of cards with matching widths
   *
   * @default 300
   */
  width?: string | number;

  /**
   * Add a row of content at the bottom of the card
   * Note this content will stretch to the edge of the card, client will need to provide custom padding as needed
   *
   * @default undefined
   */
  bottomBarContent?: React.ReactNode;

  /**
   * Add a row of content at the top of the card
   * Note this content will stretch to the edge of the card, client will need to provide custom padding as needed
   *
   * @default undefined
   */
  topBarContent?: React.ReactNode;
  // Use this to navigate to a different page via a native anchor tag
  href?: string;
  anchorProps?: HTMLProps<HTMLAnchorElement>;
  // Use this to navigate to a different page via a custom navigation function
  navigateFn?: () => Promise<void>;
}

// Loading state requires a width since it'll have no content
const LOADING_STATE_DEFAULT_WIDTH = 300;

function getStyles(args: {
  theme: Theme;
  width: CardProps['width'];
  disableHover: CardProps['disableHover'];
  loading: CardProps['loading'];
  hasBottomBar: boolean;
  hasTopBar: boolean;
  hasHref: boolean;
  useNewShadows?: boolean;
}): SerializedStyles {
  const { theme, loading, width, disableHover, hasTopBar, hasBottomBar, hasHref, useNewShadows } = args;

  const hoverOrFocusStyle = {
    boxShadow: disableHover || loading ? '' : useNewShadows ? theme.shadows.sm : theme.general.shadowLow,
    ...(hasHref && {
      border: `1px solid ${theme.colors.actionDefaultBorderHover}`,
      ...(useNewShadows && {
        boxShadow: theme.shadows.md,
      }),
    }),
  };

  return css({
    color: theme.colors.textPrimary,
    backgroundColor: theme.colors.backgroundPrimary,
    position: 'relative',
    display: 'flex',
    justifyContent: 'flex-start',
    flexDirection: 'column',
    paddingRight: hasTopBar || hasBottomBar ? 0 : theme.spacing.md,
    paddingLeft: hasTopBar || hasBottomBar ? 0 : theme.spacing.md,
    paddingTop: hasTopBar ? 0 : theme.spacing.md,
    paddingBottom: hasBottomBar ? 0 : theme.spacing.md,
    width: width ?? 'fit-content',
    borderRadius: theme.legacyBorders.borderRadiusMd,
    borderColor: theme.colors.border,
    borderWidth: '1px',
    borderStyle: 'solid',
    '&:hover': hoverOrFocusStyle,
    '&:focus': hoverOrFocusStyle,
    cursor: disableHover || loading ? 'default' : 'pointer',
    ...(useNewShadows && {
      boxShadow: theme.shadows.sm,
    }),
    transition: `box-shadow 0.2s ease-in-out`,
    textDecoration: 'none !important',
    ...getAnimationCss(theme.options.enableAnimation),
  });
}

function getBottomBarStyles(theme: Theme) {
  return css({
    marginTop: theme.spacing.sm,
    borderBottomRightRadius: theme.legacyBorders.borderRadiusMd,
    borderBottomLeftRadius: theme.legacyBorders.borderRadiusMd,
    overflow: 'hidden',
  });
}

function getTopBarStyles(theme: Theme) {
  return css({
    marginBottom: theme.spacing.sm,
    borderTopRightRadius: theme.legacyBorders.borderRadiusMd,
    borderTopLeftRadius: theme.legacyBorders.borderRadiusMd,
    overflow: 'hidden',
  });
}

export const Card = ({
  children,
  customLoadingContent,
  dangerouslyAppendEmotionCSS,
  loading,
  width,
  bottomBarContent,
  topBarContent,
  disableHover,
  onClick,
  href,
  navigateFn,
  anchorProps,
  componentId,
  analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnClick],
  shouldStartInteraction,
  ...dataAndAttributes
}: PropsWithChildren<CardProps>) => {
  const { theme } = useDesignSystemTheme();
  const { useNewShadows } = useDesignSystemSafexFlags();
  const hasTopBar = !isUndefined(topBarContent);
  const hasBottomBar = !isUndefined(bottomBarContent);
  const cardStyle = css(
    getStyles({ theme, loading, width, disableHover, hasBottomBar, hasTopBar, hasHref: Boolean(href), useNewShadows }),
  );
  const ref = React.useRef<HTMLDivElement>(null);

  const bottomBar = bottomBarContent ? <div css={css(getBottomBarStyles(theme))}>{bottomBarContent}</div> : null;
  const topBar = topBarContent ? <div css={css(getTopBarStyles(theme))}>{topBarContent}</div> : null;
  const contentPadding = hasTopBar || hasBottomBar ? theme.spacing.lg : 0;
  const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.Card,
    componentId,
    analyticsEvents: memoizedAnalyticsEvents,
    shouldStartInteraction,
  });

  const { elementRef: cardRef } = useNotifyOnFirstView<HTMLDivElement>({ onView: eventContext.onView });
  const mergedRef = useMergeRefs<HTMLDivElement>([ref, cardRef]);

  const navigate = useCallback(async () => {
    if (navigateFn) {
      await navigateFn();
    }
  }, [navigateFn]);

  const handleClick = useCallback(
    async (e: React.MouseEvent) => {
      eventContext.onClick(e);
      await navigate();
      onClick?.(e);
      ref.current?.blur();
    },
    [navigate, eventContext, onClick],
  );

  const handleSelection = useCallback(
    async (e: React.KeyboardEvent) => {
      eventContext.onClick(e);
      e.preventDefault();
      await navigate();
      onClick?.(e);
    },
    [navigate, eventContext, onClick],
  );

  const content = (
    <div
      ref={mergedRef}
      // eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex
      tabIndex={0}
      {...addDebugOutlineIfEnabled()}
      css={href ? [] : [cardStyle, dangerouslyAppendEmotionCSS]}
      onClick={loading || href ? undefined : handleClick}
      {...dataAndAttributes}
      {...(href && {
        role: 'link',
        onKeyDown: async (e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            await handleSelection(e);
          }
          dataAndAttributes.onKeyDown?.(e);
        },
      })}
      {...eventContext.dataComponentProps}
    >
      {loading ? (
        <DefaultCardLoadingContent width={width} customLoadingContent={customLoadingContent} />
      ) : (
        <>
          {topBar}
          <div css={{ padding: `0px ${contentPadding}px`, flexGrow: 1 }}>{children}</div>
          {bottomBar}
        </>
      )}
    </div>
  );

  return href ? (
    <a css={[cardStyle, dangerouslyAppendEmotionCSS]} href={href} {...anchorProps}>
      {content}
    </a>
  ) : (
    content
  );
};

function DefaultCardLoadingContent({ customLoadingContent, width }: Pick<CardProps, 'customLoadingContent' | 'width'>) {
  if (customLoadingContent) {
    return <>{customLoadingContent}</>;
  }
  return (
    <div css={{ width: width ?? LOADING_STATE_DEFAULT_WIDTH }}>
      <TitleSkeleton label="Loading..." style={{ width: '50%' }} />
      {[...Array(3).keys()].map((i) => (
        <ParagraphSkeleton label="Loading..." key={i} />
      ))}
    </div>
  );
}
