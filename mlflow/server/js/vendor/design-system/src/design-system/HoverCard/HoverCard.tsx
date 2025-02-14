import type { CSSObject } from '@emotion/react';
import * as RadixHoverCard from '@radix-ui/react-hover-card';
import type {
  HoverCardContentProps as RadixHoverCardContentProps,
  HoverCardProps as RadixHoverCardRootProps,
} from '@radix-ui/react-hover-card';
import React from 'react';

import { useDesignSystemTheme } from '../Hooks';
import { useDesignSystemContext } from '../Hooks/useDesignSystemContext';
import { useDesignSystemSafexFlags } from '../utils';
import { getDarkModePortalStyles, importantify } from '../utils/css-utils';
/**
 * Props for the HoverCard component.
 * This interface extends specific props from Radix HoverCard components to maintain flexibility and alignment
 * with Radix's capabilities while providing a simpler API for end-users
 */
export interface HoverCardProps
  extends Pick<
      RadixHoverCardContentProps,
      'side' | 'sideOffset' | 'align' | 'alignOffset' | 'avoidCollisions' | 'collisionPadding' | 'sticky'
    >,
    Pick<RadixHoverCardRootProps, 'defaultOpen' | 'open' | 'onOpenChange' | 'openDelay' | 'closeDelay'> {
  trigger: React.ReactNode;
  content: React.ReactNode;
  minWidth?: CSSObject['minWidth'];
  maxWidth?: CSSObject['maxWidth'];
}
/**
 * The HoverCard component combines Radix's HoverCard primitives into a single, easy-to-use component.
 * It handles the setup of the trigger, content, and arrow elements, as well as applying custom styles
 * using Emotion CSS
 */
export const HoverCard: React.FC<HoverCardProps> = ({
  trigger,
  content,
  side = 'top',
  sideOffset = 4,
  align = 'center',
  minWidth = 220,
  maxWidth,
  ...props
}) => {
  const { getPopupContainer } = useDesignSystemContext();
  const hoverCardStyles = useHoverCardStyles({ minWidth, maxWidth });
  return (
    <RadixHoverCard.Root {...props}>
      <RadixHoverCard.Trigger asChild>{trigger}</RadixHoverCard.Trigger>
      <RadixHoverCard.Portal container={getPopupContainer && getPopupContainer()}>
        <RadixHoverCard.Content side={side} sideOffset={sideOffset} align={align} css={hoverCardStyles['content']}>
          {content}
          <RadixHoverCard.Arrow css={hoverCardStyles['arrow']} />
        </RadixHoverCard.Content>
      </RadixHoverCard.Portal>
    </RadixHoverCard.Root>
  );
};
// CONSTANTS used for defining the Arrow's appearance and behavior
const CONSTANTS = {
  arrowWidth: 12,
  arrowHeight: 6,
  arrowBottomLength() {
    // The built in arrow is a polygon: 0,0 30,0 15,10
    return 30;
  },
  arrowSide() {
    return 2 * (this.arrowHeight ** 2 * 2) ** 0.5;
  },
  arrowStrokeWidth() {
    // This is eyeballed b/c relative to the svg viewbox coordinate system
    return 2;
  },
};
/**
 * A custom hook to generate CSS styles for the HoverCard's content and arrow.
 * These styles are dynamically generated based on the theme and optional min/max width props.
 * The hook also applies necessary dark mode adjustments
 */
const useHoverCardStyles = ({
  minWidth,
  maxWidth,
}: {
  minWidth?: CSSObject['minWidth'];
  maxWidth?: CSSObject['maxWidth'];
}): Record<string, CSSObject> => {
  const { theme } = useDesignSystemTheme();
  const { useNewShadows } = useDesignSystemSafexFlags();

  return {
    content: {
      backgroundColor: theme.colors.backgroundPrimary,
      color: theme.colors.textPrimary,
      lineHeight: theme.typography.lineHeightBase,
      border: `1px solid ${theme.colors.borderDecorative}`,
      borderRadius: theme.legacyBorders.borderRadiusMd,
      padding: `${theme.spacing.sm}px`,
      boxShadow: useNewShadows ? theme.shadows.lg : theme.general.shadowLow,
      userSelect: 'none',
      zIndex: theme.options.zIndexBase + 30,
      minWidth,
      maxWidth,
      ...getDarkModePortalStyles(theme, useNewShadows),
      a: importantify({
        color: theme.colors.actionTertiaryTextDefault,
        cursor: 'default',
        '&:hover, &:focus': {
          color: theme.colors.actionTertiaryTextHover,
        },
      }),
      '&:focus-visible': {
        outlineStyle: 'solid',
        outlineWidth: '2px',
        outlineOffset: '1px',
        outlineColor: theme.colors.actionDefaultBorderFocus,
      },
    },
    arrow: {
      fill: theme.colors.backgroundPrimary,
      height: CONSTANTS.arrowHeight,
      stroke: theme.colors.borderDecorative,
      strokeDashoffset: -CONSTANTS.arrowBottomLength(),
      strokeDasharray: CONSTANTS.arrowBottomLength() + 2 * CONSTANTS.arrowSide(),
      strokeWidth: CONSTANTS.arrowStrokeWidth(),
      width: CONSTANTS.arrowWidth,
      position: 'relative',
      top: -1,
      zIndex: theme.options.zIndexBase + 30,
    },
  };
};
