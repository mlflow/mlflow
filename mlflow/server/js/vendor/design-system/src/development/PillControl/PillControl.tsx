import type { Interpolation } from '@emotion/react';
import type { RadioGroupProps, RadioGroupItemProps } from '@radix-ui/react-radio-group';
import { RadioGroup, RadioGroupItem } from '@radix-ui/react-radio-group';
import React, { useCallback, useMemo } from 'react';

import {
  DesignSystemEventProviderAnalyticsEventTypes,
  DesignSystemEventProviderComponentTypes,
  useDesignSystemEventComponentCallbacks,
  useDesignSystemSafexFlags,
  useDesignSystemTheme,
} from '../../design-system';
import type { AnalyticsEventValueChangeNoPiiFlagProps } from '../../design-system/types';
import type { Theme } from '../../theme';
type RadioGroupSize = 'small' | 'medium' | 'large';
const RadioGroupContext = React.createContext<RadioGroupSize>('medium');
interface RootProps
  extends Pick<
      RadioGroupProps,
      'defaultValue' | 'value' | 'onValueChange' | 'disabled' | 'name' | 'required' | 'children'
    >,
    AnalyticsEventValueChangeNoPiiFlagProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange> {
  size?: RadioGroupSize;
}
export const Root = React.forwardRef<HTMLDivElement, RootProps>(
  (
    {
      size,
      componentId,
      analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange],
      valueHasNoPii,
      onValueChange,
      ...props
    },
    forwardedRef,
  ) => {
    const { theme } = useDesignSystemTheme();
    const contextValue = React.useMemo(() => size ?? 'medium', [size]);
    const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
    const eventContext = useDesignSystemEventComponentCallbacks({
      componentType: DesignSystemEventProviderComponentTypes.PillControl,
      componentId,
      analyticsEvents: memoizedAnalyticsEvents,
      valueHasNoPii,
    });
    const onValueChangeWrapper = useCallback(
      (value: string) => {
        eventContext.onValueChange?.(value);
        onValueChange?.(value);
      },
      [eventContext, onValueChange],
    );
    return (
      <RadioGroupContext.Provider value={contextValue}>
        <RadioGroup
          css={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: theme.spacing.sm,
          }}
          onValueChange={onValueChangeWrapper}
          {...props}
          ref={forwardedRef}
          {...eventContext.dataComponentProps}
        />
      </RadioGroupContext.Provider>
    );
  },
);
interface ItemProps extends Pick<RadioGroupItemProps, 'children' | 'value' | 'disabled' | 'required'> {
  icon?: React.ReactNode;
}
export const Item = React.forwardRef<HTMLButtonElement, ItemProps>(({ children, icon, ...props }, forwardedRef) => {
  const size = React.useContext(RadioGroupContext);
  const { theme } = useDesignSystemTheme();
  const iconClass = 'pill-control-icon';
  const css = useRadioGroupItemStyles(size, iconClass);
  return (
    <RadioGroupItem css={css} {...props}>
      {icon && (
        <span
          className={iconClass}
          css={{
            marginRight: size === 'large' ? theme.spacing.sm : theme.spacing.xs,
            [`& > .anticon`]: { verticalAlign: `-3px` },
          }}
        >
          {icon}
        </span>
      )}
      {children}
    </RadioGroupItem>
  );
});
const useRadioGroupItemStyles = (size: RadioGroupSize, iconClass: string): Interpolation<Theme> => {
  const { theme } = useDesignSystemTheme();
  const { useNewShadows } = useDesignSystemSafexFlags();

  return {
    textOverflow: 'ellipsis',
    ...(useNewShadows
      ? {
          boxShadow: theme.shadows.xs,
        }
      : {}),
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    appearance: 'none',
    textDecoration: 'none',
    background: 'none',
    border: '1px solid',
    cursor: 'pointer',
    backgroundColor: theme.colors.actionDefaultBackgroundDefault,
    borderColor: theme.colors.border,
    color: theme.colors.textPrimary,
    lineHeight: theme.typography.lineHeightBase,
    height: 32,
    paddingInline: '12px',
    fontWeight: theme.typography.typographyRegularFontWeight,
    fontSize: theme.typography.fontSizeBase,
    borderRadius: theme.spacing.md,
    transition: 'background-color 0.2s ease-in-out, border-color 0.2s ease-in-out',
    [`& > .${iconClass}`]: {
      color: theme.colors.textSecondary,
      ...(size === 'large'
        ? {
            backgroundColor: theme.colors.tagDefault,
            padding: theme.spacing.sm,
            borderRadius: theme.spacing.md,
          }
        : {}),
    },
    '&[data-state="checked"]': {
      backgroundColor: theme.colors.actionDefaultBackgroundPress,
      borderColor: 'transparent',
      color: theme.colors.textPrimary,
      // outline
      outlineStyle: 'solid',
      outlineWidth: '2px',
      outlineOffset: '0px',
      outlineColor: theme.colors.actionDefaultBorderFocus,
      '&:hover': {
        backgroundColor: theme.colors.actionDefaultBackgroundPress,
        borderColor: theme.colors.actionLinkPress,
        color: 'inherit',
      },
      [`& > .${iconClass}, &:hover > .${iconClass}`]: {
        color: theme.colors.actionDefaultTextPress,
        ...(size === 'large'
          ? {
              backgroundColor: theme.colors.actionIconBackgroundPress,
            }
          : {}),
      },
    },
    '&:focus-visible': {
      outlineStyle: 'solid',
      outlineWidth: '2px',
      outlineOffset: '0px',
      outlineColor: theme.colors.actionDefaultBorderFocus,
    },
    '&:hover': {
      backgroundColor: theme.colors.actionDefaultBackgroundHover,
      borderColor: theme.colors.actionLinkHover,
      color: theme.colors.actionDefaultTextHover,
      [`& > .${iconClass}`]: {
        color: 'inherit',
        ...(size === 'large'
          ? {
              backgroundColor: theme.colors.actionIconBackgroundHover,
            }
          : {}),
      },
    },
    '&:active': {
      backgroundColor: theme.colors.actionDefaultBackgroundPress,
      borderColor: theme.colors.actionLinkPress,
      color: theme.colors.actionDefaultTextPress,
      [`& > .${iconClass}`]: {
        color: 'inherit',
        ...(size === 'large'
          ? {
              backgroundColor: theme.colors.actionIconBackgroundPress,
            }
          : {}),
      },
    },
    '&:disabled': {
      backgroundColor: theme.colors.actionDisabledBackground,
      borderColor: theme.colors.actionDisabledBorder,
      color: theme.colors.actionDisabledText,
      cursor: 'not-allowed',
      [`& > .${iconClass}`]: {
        color: 'inherit',
      },
    },
    ...(size === 'small'
      ? {
          height: 24,
          lineHeight: theme.typography.lineHeightSm,
          paddingInline: theme.spacing.sm,
        }
      : {}),
    ...(size === 'large'
      ? {
          height: 44,
          lineHeight: theme.typography.lineHeightXl,
          paddingInline: theme.spacing.md,
          paddingInlineStart: '6px',
          borderRadius: theme.spacing.lg,
        }
      : {}),
  };
};
