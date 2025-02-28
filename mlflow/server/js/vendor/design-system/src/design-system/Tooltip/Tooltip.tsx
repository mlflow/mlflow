import type { CSSObject } from '@emotion/react';
import { keyframes } from '@emotion/react';
import * as RadixTooltip from '@radix-ui/react-tooltip';
import type {
  TooltipContentProps as RadixTooltipContentProps,
  TooltipProps as RadixTooltipRootProps,
} from '@radix-ui/react-tooltip';
import type { HTMLAttributes } from 'react';
import React, { useCallback, useMemo, useRef } from 'react';

import {
  DesignSystemEventProviderAnalyticsEventTypes,
  DesignSystemEventProviderComponentTypes,
  useDesignSystemEventComponentCallbacks,
} from '../DesignSystemEventProvider';
import { useDesignSystemTheme } from '../Hooks';
import { useDesignSystemContext } from '../Hooks/useDesignSystemContext';
import type { AnalyticsEventProps, HTMLDataAttributes } from '../types';
import { useDesignSystemSafexFlags } from '../utils';
import { isInsideTest } from '../utils/test-utils';

export interface TooltipProps
  extends HTMLDataAttributes,
    Pick<
      RadixTooltipContentProps,
      | 'side'
      | 'sideOffset'
      | 'align'
      | 'alignOffset'
      | 'avoidCollisions'
      | 'collisionPadding'
      | 'sticky'
      | 'hideWhenDetached'
    >,
    Pick<RadixTooltipRootProps, 'defaultOpen' | 'delayDuration'>,
    AnalyticsEventProps<DesignSystemEventProviderAnalyticsEventTypes.OnView> {
  /**
   * The element with which the tooltip should be associated.
   */
  children: React.ReactNode;
  /**
   * Content or text that will appear within the tooltip.
   */
  content: React.ReactNode;
  /**
   * Override for the default max width of the tooltip.
   */
  maxWidth?: CSSObject['maxWidth'];
  /**
   * Override for the default z-index of the tooltip.
   */
  zIndex?: number;
}

const useTooltipStyles = ({ maxWidth }: { maxWidth: CSSObject['maxWidth'] }): Record<string, CSSObject> => {
  const { theme, classNamePrefix: clsPrefix } = useDesignSystemTheme();
  const { useNewShadows } = useDesignSystemSafexFlags();

  const classTypography = `.${clsPrefix}-typography`;
  const { isDarkMode } = theme;
  const slideUpAndFade = keyframes({
    from: {
      opacity: 0,
      transform: 'translateY(2px)',
    },
    to: {
      opacity: 1,
      transform: 'translateY(0)',
    },
  });
  const slideRightAndFade = keyframes({
    from: {
      opacity: 0,
      transform: 'translateX(-2px)',
    },
    to: {
      opacity: 1,
      transform: 'translateX(0)',
    },
  });
  const slideDownAndFade = keyframes({
    from: {
      opacity: 0,
      transform: 'translateY(-2px)',
    },
    to: {
      opacity: 1,
      transform: 'translateY(0)',
    },
  });
  const slideLeftAndFade = keyframes({
    from: {
      opacity: 0,
      transform: 'translateX(2px)',
    },
    to: {
      opacity: 1,
      transform: 'translateX(0)',
    },
  });

  const linkColor = isDarkMode ? theme.colors.blue600 : theme.colors.blue500;
  const linkActiveColor = isDarkMode ? theme.colors.blue800 : theme.colors.blue300;
  const linkHoverColor = isDarkMode ? theme.colors.blue700 : theme.colors.blue400;

  return {
    content: {
      backgroundColor: theme.colors.tooltipBackgroundTooltip,
      color: theme.colors.actionPrimaryTextDefault,
      borderRadius: theme.legacyBorders.borderRadiusMd,
      fontSize: theme.typography.fontSizeMd,
      padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
      lineHeight: theme.typography.lineHeightLg,
      fontWeight: theme.typography.typographyRegularFontWeight,
      boxShadow: useNewShadows ? theme.shadows.lg : theme.general.shadowHigh,
      maxWidth: maxWidth,
      wordWrap: 'break-word',
      whiteSpace: 'normal',
      zIndex: theme.options.zIndexBase + 70,
      willChange: 'transform, opacity',
      "&[data-state='delayed-open'][data-side='top']": {
        animation: `${slideDownAndFade} 400ms cubic-bezier(0.16, 1, 0.3, 1)`,
      },
      "&[data-state='delayed-open'][data-side='right']": {
        animation: `${slideLeftAndFade} 400ms cubic-bezier(0.16, 1, 0.3, 1)`,
      },
      "&[data-state='delayed-open'][data-side='bottom']": {
        animation: `${slideUpAndFade} 400ms cubic-bezier(0.16, 1, 0.3, 1)`,
      },
      "&[data-state='delayed-open'][data-side='left']": {
        animation: `${slideRightAndFade} 400ms cubic-bezier(0.16, 1, 0.3, 1)`,
      },
      [`& a${classTypography}`]: {
        '&, :focus': {
          color: linkColor,
          '.anticon': { color: linkColor },
        },
        ':active': {
          color: linkActiveColor,
          '.anticon': { color: linkActiveColor },
        },
        ':hover': {
          color: linkHoverColor,
          '.anticon': { color: linkHoverColor },
        },
      },
    },
    arrow: {
      fill: theme.colors.tooltipBackgroundTooltip,
      zIndex: theme.options.zIndexBase + 70,
      visibility: 'visible',
    },
  };
};

/**
 * If the tooltip is not displaying for you, it might be because the child does not accept the onMouseEnter, onMouseLeave, onPointerEnter,
 * onPointerLeave, onFocus, and onClick props. You can add these props to your child component, or wrap it in a `<span>` tag.
 *
 * See go/dubois.
 */
export const Tooltip: React.FC<TooltipProps> & HTMLAttributes<HTMLDivElement> = ({
  children,
  content,
  defaultOpen = false,
  delayDuration = 350,
  side = 'top',
  sideOffset = 4,
  align = 'center',
  maxWidth = 250,
  componentId,
  analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnView],
  zIndex,
  ...props
}) => {
  const { theme } = useDesignSystemTheme();
  const { getPopupContainer } = useDesignSystemContext();
  const tooltipStyles = useTooltipStyles({ maxWidth });

  const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.Tooltip,
    componentId,
    analyticsEvents: memoizedAnalyticsEvents,
  });
  const firstView = useRef(true);
  const handleOpenChange = useCallback(
    (open: boolean) => {
      if (open && firstView.current) {
        eventContext.onView();
        firstView.current = false;
      }
    },
    [eventContext, firstView],
  );

  const IS_INSIDE_TEST = isInsideTest();

  return (
    <RadixTooltip.Root
      defaultOpen={defaultOpen}
      delayDuration={IS_INSIDE_TEST ? 10 : delayDuration}
      onOpenChange={handleOpenChange}
    >
      <RadixTooltip.Trigger asChild>{children}</RadixTooltip.Trigger>
      {content ? (
        <RadixTooltip.Portal container={getPopupContainer && getPopupContainer()}>
          <RadixTooltip.Content
            side={side}
            align={align}
            sideOffset={theme.spacing.sm}
            arrowPadding={theme.spacing.md}
            css={[tooltipStyles['content'], zIndex ? { zIndex } : undefined]}
            {...props}
            {...eventContext.dataComponentProps}
          >
            {content}
            <RadixTooltip.Arrow css={tooltipStyles['arrow']} />
          </RadixTooltip.Content>
        </RadixTooltip.Portal>
      ) : null}
    </RadixTooltip.Root>
  );
};
