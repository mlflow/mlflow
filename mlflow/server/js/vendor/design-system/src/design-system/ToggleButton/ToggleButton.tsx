import { css } from '@emotion/react';
import * as Toggle from '@radix-ui/react-toggle';
import type { Ref } from 'react';
import React, { forwardRef, useCallback, useEffect, useMemo } from 'react';

import type { Theme } from '../../theme';
import type { ButtonSize } from '../Button';
import {
  DesignSystemEventProviderAnalyticsEventTypes,
  DesignSystemEventProviderComponentTypes,
  useDesignSystemEventComponentCallbacks,
} from '../DesignSystemEventProvider';
import { useDesignSystemTheme } from '../Hooks';
import type { IconProps } from '../Icon';
import { CheckIcon, Icon } from '../Icon';
import type { AnalyticsEventProps } from '../types';
import { useDesignSystemSafexFlags } from '../utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';

const SMALL_BUTTON_HEIGHT = 24;

interface ToggleProps
  extends React.ComponentProps<typeof Toggle.Root>,
    AnalyticsEventProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange> {
  icon?: React.ReactNode;
  size?: ButtonSize;
}

const getStyles = (theme: Theme, size: ButtonSize, onlyIcon: boolean, useNewShadows: boolean) => {
  return css({
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    whiteSpace: 'nowrap',
    ...(useNewShadows && {
      boxShadow: theme.shadows.xs,
    }),

    border: `1px solid ${theme.colors.actionDefaultBorderDefault}`,
    borderRadius: theme.general.borderRadiusBase,

    backgroundColor: 'transparent',
    color: theme.colors.actionDefaultTextDefault,

    height: theme.general.heightSm,
    padding: '0 12px',

    fontSize: theme.typography.fontSizeBase,
    lineHeight: `${theme.typography.lineHeightBase}px`,

    '&[data-state="off"] .togglebutton-icon-wrapper': {
      color: theme.colors.textSecondary,
    },

    '&[data-state="off"]:hover .togglebutton-icon-wrapper': {
      color: theme.colors.actionDefaultTextHover,
    },

    '&[data-state="on"]': {
      backgroundColor: theme.colors.actionDefaultBackgroundPress,
      color: theme.colors.actionDefaultTextPress,
      borderColor: theme.colors.actionDefaultBorderPress,
    },

    '&:hover': {
      cursor: 'pointer',
      color: theme.colors.actionDefaultTextHover,
      backgroundColor: theme.colors.actionDefaultBackgroundHover,
      borderColor: theme.colors.actionDefaultBorderHover,

      '& > svg': {
        stroke: theme.colors.actionDefaultBorderHover,
      },
    },

    '&:disabled': {
      cursor: 'default',
      borderColor: theme.colors.actionDisabledBorder,
      color: theme.colors.actionDisabledText,
      backgroundColor: 'transparent',
      ...(useNewShadows && {
        boxShadow: 'none',
      }),

      '& > svg': {
        stroke: theme.colors.border,
      },
    },

    ...(!onlyIcon && {
      '&&': {
        padding: '4px 12px',

        ...(size === 'small' && {
          padding: '0 8px',
        }),
      },
    }),

    ...(onlyIcon && {
      width: theme.general.heightSm,
      border: 'none',
    }),

    ...(size === 'small' && {
      height: SMALL_BUTTON_HEIGHT,
      lineHeight: theme.typography.lineHeightBase,

      ...(onlyIcon && {
        width: SMALL_BUTTON_HEIGHT,
        paddingTop: 0,
        paddingBottom: 0,
        verticalAlign: 'middle',
      }),
    }),
  });
};

const RectangleSvg = (props: React.SVGAttributes<any>) => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" {...props}>
    <rect x="0.5" y="0.5" width="15" height="15" rx="3.5" />
  </svg>
);
const RectangleIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={RectangleSvg} />;
});

export const ToggleButton = forwardRef<HTMLButtonElement, ToggleProps>(
  (
    {
      children,
      pressed,
      defaultPressed,
      icon,
      size = 'middle',
      componentId,
      analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange],
      ...props
    },
    ref,
  ) => {
    const { theme } = useDesignSystemTheme();
    const { useNewShadows } = useDesignSystemSafexFlags();
    const [isPressed, setIsPressed] = React.useState(defaultPressed);

    const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
    const eventContext = useDesignSystemEventComponentCallbacks({
      componentType: DesignSystemEventProviderComponentTypes.ToggleButton,
      componentId: componentId,
      analyticsEvents: memoizedAnalyticsEvents,
      valueHasNoPii: true,
    });

    const handleOnPressedChange = useCallback(
      (pressed: boolean) => {
        eventContext.onValueChange(pressed);
        props.onPressedChange?.(pressed);
        setIsPressed(pressed);
      },
      [eventContext, props],
    );

    useEffect(() => {
      setIsPressed(pressed);
    }, [pressed]);

    const iconOnly = !children && Boolean(icon);
    const iconStyle = iconOnly ? {} : { marginRight: theme.spacing.xs };

    const checkboxIcon = isPressed ? (
      <CheckIcon />
    ) : (
      <RectangleIcon
        css={{
          stroke: theme.colors.border,
        }}
      />
    );

    return (
      <Toggle.Root
        {...addDebugOutlineIfEnabled()}
        css={getStyles(theme, size, iconOnly, useNewShadows)}
        {...props}
        pressed={isPressed}
        onPressedChange={handleOnPressedChange}
        ref={ref}
        {...eventContext.dataComponentProps}
      >
        <span className="togglebutton-icon-wrapper" style={{ display: 'flex', ...iconStyle }}>
          {icon ? icon : checkboxIcon}
        </span>
        {children}
      </Toggle.Root>
    );
  },
);
