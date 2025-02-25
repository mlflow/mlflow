import type { CSSObject } from '@emotion/react';
import type { HTMLAttributes } from 'react';
import React, { useCallback, useMemo } from 'react';

import { useDesignSystemSafexFlags, useDesignSystemTheme } from '../../design-system';
import type { DesignSystemEventProviderAnalyticsEventTypes } from '../../design-system/DesignSystemEventProvider/DesignSystemEventProvider';
import {
  DesignSystemEventProviderComponentTypes,
  useDesignSystemEventComponentCallbacks,
} from '../../design-system/DesignSystemEventProvider/DesignSystemEventProvider';
import type { AnalyticsEventOptionalProps, DangerousGeneralProps } from '../../design-system/types';
import { addDebugOutlineIfEnabled } from '../../design-system/utils/debug';
export interface PreviewCardProps
  extends DangerousGeneralProps,
    Omit<HTMLAttributes<HTMLDivElement>, 'title'>,
    AnalyticsEventOptionalProps<DesignSystemEventProviderAnalyticsEventTypes.OnClick> {
  icon?: React.ReactNode;
  title?: React.ReactNode;
  subtitle?: React.ReactNode;
  titleActions?: React.ReactNode;
  startActions?: React.ReactNode;
  endActions?: React.ReactNode;
  image?: React.ReactNode;
  size?: 'default' | 'large';
  onClick?: React.MouseEventHandler<HTMLDivElement>;
  disabled?: boolean;
  selected?: boolean;
}
export const PreviewCard = ({
  icon,
  title,
  subtitle,
  titleActions,
  children,
  startActions,
  endActions,
  image,
  onClick,
  size = 'default',
  dangerouslyAppendEmotionCSS,
  componentId,
  analyticsEvents = [],
  disabled,
  selected,
  ...props
}: PreviewCardProps) => {
  const styles = usePreviewCardStyles({ onClick, size, disabled });
  const tabIndex = onClick ? 0 : undefined;
  const role = onClick ? 'button' : undefined;
  const showFooter = startActions || endActions;
  const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.PreviewCard,
    componentId,
    analyticsEvents: memoizedAnalyticsEvents,
  });
  const onClickWrapper = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (onClick) {
        eventContext.onClick(e);
        onClick(e);
      }
    },
    [eventContext, onClick],
  );
  return (
    <div
      {...addDebugOutlineIfEnabled()}
      css={[styles['container'], dangerouslyAppendEmotionCSS]}
      tabIndex={tabIndex}
      onClick={onClickWrapper}
      onKeyDown={(e) => {
        if (!onClick || disabled) {
          return;
        }

        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onClickWrapper(e as unknown as React.MouseEvent<HTMLDivElement>);
        }
      }}
      role={role}
      aria-disabled={disabled}
      aria-pressed={selected}
      {...props}
    >
      {image && <div css={styles['image']}>{image}</div>}
      <div css={styles['header']}>
        {icon && <div>{icon}</div>}
        <div css={styles['titleWrapper']}>
          {title && <div css={styles['title']}>{title}</div>}
          {subtitle && <div css={styles['subTitle']}>{subtitle}</div>}
        </div>
        {titleActions && <div>{titleActions}</div>}
      </div>
      {children && <div css={styles['childrenWrapper']}>{children}</div>}
      {showFooter && (
        <div css={styles['footer']}>
          <div css={styles['action']}>{startActions}</div>
          <div css={styles['action']}>{endActions}</div>
        </div>
      )}
    </div>
  );
};
const usePreviewCardStyles = ({
  onClick,
  size,
  disabled,
}: Pick<PreviewCardProps, 'onClick' | 'size' | 'disabled'>): Record<string, CSSObject> => {
  const { theme } = useDesignSystemTheme();
  const { useNewShadows } = useDesignSystemSafexFlags();
  const isInteractive = onClick !== undefined;
  return {
    container: {
      borderRadius: theme.legacyBorders.borderRadiusLg,
      border: `1px solid ${theme.colors.border}`,
      padding: size === 'large' ? theme.spacing.lg : theme.spacing.md,
      color: theme.colors.textSecondary,
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'space-between',
      gap: size === 'large' ? theme.spacing.md : theme.spacing.sm,
      ...(useNewShadows
        ? {
            boxShadow: theme.shadows.sm,
          }
        : {}),
      cursor: isInteractive ? 'pointer' : 'default',
      ...(isInteractive && {
        '&[aria-disabled="true"]': {
          pointerEvents: 'none',
          backgroundColor: theme.colors.actionDisabledBackground,
          borderColor: theme.colors.actionDisabledBorder,
          color: theme.colors.actionDisabledText,
        },
        '&:hover, &:focus-within': {
          boxShadow: useNewShadows ? theme.shadows.md : theme.general.shadowLow,
        },
        '&:active': {
          background: theme.colors.actionTertiaryBackgroundPress,
          borderColor: theme.colors.actionDefaultBorderHover,
          boxShadow: useNewShadows ? theme.shadows.md : theme.general.shadowLow,
        },
        '&:focus, &[aria-pressed="true"]': {
          outlineColor: theme.colors.actionDefaultBorderFocus,
          outlineWidth: 2,
          outlineOffset: -2,
          outlineStyle: 'solid',
          boxShadow: useNewShadows ? theme.shadows.md : theme.general.shadowLow,
          borderColor: theme.colors.actionDefaultBorderHover,
        },
        '&:active:not(:focus):not(:focus-within)': {
          background: 'transparent',
          borderColor: theme.colors.border,
        },
      }),
    },
    image: {
      '& > *': {
        borderRadius: theme.legacyBorders.borderRadiusMd,
      },
    },
    header: {
      display: 'flex',
      alignItems: 'center',
      gap: theme.spacing.sm,
    },
    title: {
      fontWeight: theme.typography.typographyBoldFontWeight,
      color: disabled ? theme.colors.actionDisabledText : theme.colors.textPrimary,
      lineHeight: theme.typography.lineHeightSm,
    },
    subTitle: {
      lineHeight: theme.typography.lineHeightSm,
    },
    titleWrapper: {
      flexGrow: 1,
      overflow: 'hidden',
    },
    childrenWrapper: {
      flexGrow: 1,
    },
    footer: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      flexWrap: 'wrap',
    },
    action: {
      overflow: 'hidden',
      // to ensure focus ring is rendered
      margin: theme.spacing.md * -1,
      padding: theme.spacing.md,
    },
  };
};
