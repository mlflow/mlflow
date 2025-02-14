import type { CSSObject, Interpolation, Theme } from '@emotion/react';
import * as Popover from '@radix-ui/react-popover';
import type { ReactElement } from 'react';
import { forwardRef, useCallback, useEffect, useMemo, useRef } from 'react';

import {
  ComponentFinderContext,
  DesignSystemEventProviderAnalyticsEventTypes,
  DesignSystemEventProviderComponentTypes,
  useComponentFinderContext,
  useDesignSystemEventComponentCallbacks,
} from '../DesignSystemEventProvider';
import { useDesignSystemTheme } from '../Hooks';
import { useDesignSystemContext } from '../Hooks/useDesignSystemContext';
import type { AnalyticsEventProps } from '../types';
import { useDesignSystemSafexFlags } from '../utils';
import { getDarkModePortalStyles, importantify } from '../utils/css-utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';

// WRAPPED RADIX-UI-COMPONENTS
export const Anchor = Popover.Anchor; // Behavioral component only

export type PopoverRootProps = Popover.PopoverProps &
  AnalyticsEventProps<DesignSystemEventProviderAnalyticsEventTypes.OnView>;

export const Root = ({
  children,
  onOpenChange,
  componentId,
  analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnView],
  ...props
}: PopoverRootProps): ReactElement => {
  const firstView = useRef(true);
  const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.Popover,
    componentId: componentId ?? 'design_system.popover',
    analyticsEvents: memoizedAnalyticsEvents,
  });
  useEffect(() => {
    if (props.open && firstView.current) {
      eventContext.onView();
      firstView.current = false;
    }
  }, [eventContext, props.open]);
  const onOpenChangeHandler = useCallback(
    (open: boolean) => {
      if (open && firstView.current) {
        eventContext.onView();
        firstView.current = false;
      }
      onOpenChange?.(open);
    },
    [eventContext, onOpenChange],
  );
  return (
    <Popover.Root {...props} onOpenChange={onOpenChangeHandler}>
      <ComponentFinderContext.Provider value={{ dataComponentProps: eventContext.dataComponentProps }}>
        {children}
      </ComponentFinderContext.Provider>
    </Popover.Root>
  );
};

export interface PopoverProps extends Popover.PopoverContentProps {
  minWidth?: number;
  maxWidth?: number;
}

export const Content = forwardRef<HTMLDivElement, PopoverProps>(function Content(
  { children, minWidth = 220, maxWidth, ...props },
  ref,
): ReactElement {
  const { getPopupContainer } = useDesignSystemContext();
  const { theme } = useDesignSystemTheme();
  const { useNewShadows } = useDesignSystemSafexFlags();

  const dataComponentProps = useComponentFinderContext(DesignSystemEventProviderComponentTypes.Popover);

  return (
    <Popover.Portal container={getPopupContainer && getPopupContainer()}>
      <Popover.Content
        {...addDebugOutlineIfEnabled()}
        ref={ref}
        css={[contentStyles(theme, useNewShadows), { minWidth, maxWidth }]}
        sideOffset={4}
        {...props}
        {...dataComponentProps}
      >
        {children}
      </Popover.Content>
    </Popover.Portal>
  );
});

export const Trigger = forwardRef<HTMLButtonElement, Popover.PopoverTriggerProps>(function Trigger(
  { children, ...props },
  ref,
): ReactElement {
  return (
    <Popover.Trigger {...addDebugOutlineIfEnabled()} ref={ref} {...props}>
      {children}
    </Popover.Trigger>
  );
});

export const Close = forwardRef<HTMLButtonElement, Popover.PopoverCloseProps>(function Close(
  { children, ...props },
  ref,
): ReactElement {
  return (
    <Popover.Close ref={ref} {...props}>
      {children}
    </Popover.Close>
  );
});

export const Arrow = forwardRef<SVGSVGElement, Popover.PopoverArrowProps>(function Arrow(
  { children, ...props },
  ref,
): ReactElement {
  const { theme } = useDesignSystemTheme();
  return (
    <Popover.Arrow
      css={{
        fill: theme.colors.backgroundPrimary,
        stroke: theme.colors.borderDecorative,
        strokeDashoffset: -CONSTANTS.arrowBottomLength(),
        strokeDasharray: CONSTANTS.arrowBottomLength() + 2 * CONSTANTS.arrowSide(),
        strokeWidth: CONSTANTS.arrowStrokeWidth(),
        // TODO: This is a temporary fix for the alignment of the Arrow;
        // Radix has changed the implementation for v1.0.0 (uses floating-ui)
        // which has new behaviors for alignment that we don't want. Generally
        // we need to fix the arrow to always be aligned to the left of the menu (with
        // offset equal to border radius)
        position: 'relative',
        top: -1,
      }}
      ref={ref}
      width={12}
      height={6}
      {...props}
    >
      {children}
    </Popover.Arrow>
  );
});

// CONSTANTS
const CONSTANTS = {
  arrowBottomLength() {
    // The built in arrow is a polygon: 0,0 30,0 15,10
    return 30;
  },
  arrowHeight() {
    return 10;
  },
  arrowSide() {
    return 2 * (this.arrowHeight() ** 2 * 2) ** 0.5;
  },
  arrowStrokeWidth() {
    // This is eyeballed b/c relative to the svg viewbox coordinate system
    return 2;
  },
};

const popoverContentStyles = (theme: Theme, useNewShadows: boolean): CSSObject => ({
  backgroundColor: theme.colors.backgroundPrimary,
  color: theme.colors.textPrimary,
  lineHeight: theme.typography.lineHeightBase,
  border: `1px solid ${theme.colors.borderDecorative}`,
  borderRadius: theme.legacyBorders.borderRadiusMd,
  padding: `${theme.spacing.sm}px`,
  boxShadow: useNewShadows ? theme.shadows.lg : theme.general.shadowLow,
  userSelect: 'none',
  zIndex: theme.options.zIndexBase + 30,
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
});

const contentStyles = (theme: Theme, useNewShadows: boolean): Interpolation<Theme> => ({
  ...popoverContentStyles(theme, useNewShadows),
});
