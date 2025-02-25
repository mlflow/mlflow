import type { SerializedStyles } from '@emotion/react';
import { keyframes, css } from '@emotion/react';
import * as Toast from '@radix-ui/react-toast';
import React, { forwardRef, useMemo } from 'react';

import type { Theme } from '../../theme';
import { Button } from '../Button';
import {
  DesignSystemEventProviderComponentTypes,
  useDesignSystemEventComponentCallbacks,
  DesignSystemEventProviderAnalyticsEventTypes,
  DesignSystemEventProviderComponentSubTypeMap,
} from '../DesignSystemEventProvider/DesignSystemEventProvider';
import { DU_BOIS_ENABLE_ANIMATION_CLASSNAME } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks';
import { CloseIcon } from '../Icon';
import { SeverityIcon } from '../Icon/iconMap';
import type { AnalyticsEventProps } from '../types';
import { getDarkModePortalStyles, useDesignSystemSafexFlags, useNotifyOnFirstView } from '../utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';

const hideAnimation = keyframes({
  from: {
    opacity: 1,
  },
  to: {
    opacity: 0,
  },
});

const slideInAnimation = keyframes({
  from: {
    transform: 'translateX(calc(100% + 12px))',
  },
  to: {
    transform: 'translateX(0)',
  },
});

const swipeOutAnimation = keyframes({
  from: {
    transform: 'translateX(var(--radix-toast-swipe-end-x))',
  },
  to: {
    transform: 'translateX(calc(100% + 12px))',
  },
});

const getToastRootStyle = (theme: Theme, classNamePrefix: string, useNewShadows: boolean): SerializedStyles => {
  return css({
    '&&': {
      position: 'relative',
      display: 'grid',
      background: theme.colors.backgroundPrimary,
      padding: 12,
      columnGap: 4,
      boxShadow: useNewShadows ? theme.shadows.lg : theme.general.shadowLow,
      borderRadius: theme.general.borderRadiusBase,
      lineHeight: '20px',

      gridTemplateRows: '[header] auto [content] auto',
      gridTemplateColumns: '[icon] auto [content] 1fr [close] auto',
      ...getDarkModePortalStyles(theme, useNewShadows),
    },

    [`.${classNamePrefix}-notification-severity-icon`]: {
      gridRow: 'header / content',
      gridColumn: 'icon / icon',
      display: 'inline-flex',
      alignItems: 'center',
    },

    [`.${classNamePrefix}-btn`]: {
      display: 'inline-flex',
      alignItems: 'center',
      justifyContent: 'center',
    },

    [`.${classNamePrefix}-notification-info-icon`]: {
      color: theme.colors.textSecondary,
    },

    [`.${classNamePrefix}-notification-success-icon`]: {
      color: theme.colors.textValidationSuccess,
    },

    [`.${classNamePrefix}-notification-warning-icon`]: {
      color: theme.colors.textValidationWarning,
    },

    [`.${classNamePrefix}-notification-error-icon`]: {
      color: theme.colors.textValidationDanger,
    },

    '&&[data-state="open"]': {
      animation: `${slideInAnimation} 300ms cubic-bezier(0.16, 1, 0.3, 1)`,
    },

    '&[data-state="closed"]': {
      animation: `${hideAnimation} 100ms ease-in`,
    },

    '&[data-swipe="move"]': {
      transform: 'translateX(var(--radix-toast-swipe-move-x))',
    },

    '&[data-swipe="cancel"]': {
      transform: 'translateX(0)',
      transition: 'transform 200ms ease-out',
    },

    '&[data-swipe="end"]': {
      animation: `${swipeOutAnimation} 100ms ease-out`,
    },
  });
};

export interface NotificationProps
  extends Toast.ToastProps,
    AnalyticsEventProps<DesignSystemEventProviderAnalyticsEventTypes.OnView> {
  severity?: 'info' | 'success' | 'warning' | 'error';
  isCloseable?: boolean;
}

export const Root = forwardRef<HTMLLIElement, NotificationProps>(function (
  {
    children,
    severity = 'info',
    componentId,
    analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnView],
    ...props
  }: NotificationProps,
  ref,
): JSX.Element {
  const { theme, classNamePrefix } = useDesignSystemTheme();
  const { useNewShadows } = useDesignSystemSafexFlags();
  const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.Notification,
    componentId,
    componentSubType: DesignSystemEventProviderComponentSubTypeMap[severity],
    analyticsEvents: memoizedAnalyticsEvents,
    shouldStartInteraction: false,
  });
  // A new ref was created rather than creating additional complexity of merging the refs, something to consider for the future to optimize
  const { elementRef } = useNotifyOnFirstView<HTMLSpanElement>({ onView: eventContext.onView });

  return (
    <Toast.Root
      ref={ref}
      css={getToastRootStyle(theme, classNamePrefix, useNewShadows)}
      {...props}
      {...addDebugOutlineIfEnabled()}
    >
      <SeverityIcon
        className={`${classNamePrefix}-notification-severity-icon ${classNamePrefix}-notification-${severity}-icon`}
        severity={severity}
        ref={elementRef}
      />
      {children}
    </Toast.Root>
  );
});

// TODO: Support light and dark mode

const getViewportStyle = (theme: Theme): React.CSSProperties => {
  return {
    position: 'fixed',
    top: 0,
    right: 0,
    display: 'flex',
    flexDirection: 'column',
    padding: 12,
    gap: 12,
    width: 440,
    listStyle: 'none',
    zIndex: theme.options.zIndexBase + 100,
    outline: 'none',
    maxWidth: `calc(100% - ${theme.spacing.lg}px)`,
  };
};

const getTitleStyles = (theme: Theme): SerializedStyles => {
  return css({
    fontWeight: theme.typography.typographyBoldFontWeight,
    color: theme.colors.textPrimary,
    gridRow: 'header / header',
    gridColumn: 'content / content',
    userSelect: 'text',
  });
};

export interface NotificationTitleProps extends Toast.ToastTitleProps {}

export const Title = forwardRef<HTMLDivElement, NotificationTitleProps>(function (
  { children, ...props }: NotificationTitleProps,
  ref,
): JSX.Element {
  const { theme } = useDesignSystemTheme();
  return (
    <Toast.Title ref={ref} css={getTitleStyles(theme)} {...props}>
      {children}
    </Toast.Title>
  );
});

const getDescriptionStyles = (theme: Theme): SerializedStyles => {
  return css({
    marginTop: 4,
    color: theme.colors.textPrimary,
    gridRow: 'content / content',
    gridColumn: 'content / content',
    userSelect: 'text',
  });
};

export interface NotificationDescriptionProps extends Toast.ToastDescriptionProps {}

export const Description = forwardRef<HTMLDivElement, NotificationDescriptionProps>(function (
  { children, ...props }: NotificationDescriptionProps,
  ref,
): JSX.Element {
  const { theme } = useDesignSystemTheme();
  return (
    <Toast.Description ref={ref} css={getDescriptionStyles(theme)} {...props}>
      {children}
    </Toast.Description>
  );
});

const getCloseStyles = (theme: Theme): SerializedStyles => {
  return css({
    color: theme.colors.textSecondary,
    position: 'absolute',
    // Offset close button position to align with the title, title uses 20px line height, button has 32px
    right: 6,
    top: 6,
  });
};

export interface NotificationCloseProps
  extends Toast.ToastCloseProps,
    AnalyticsEventProps<DesignSystemEventProviderAnalyticsEventTypes.OnClick> {
  closeLabel?: string;
}

export const Close = forwardRef<HTMLButtonElement, NotificationCloseProps>(function (
  props: NotificationCloseProps,
  ref,
): JSX.Element {
  const { theme } = useDesignSystemTheme();

  const { closeLabel, componentId, analyticsEvents, ...restProps } = props;

  return (
    // Wrapper to keep close column width for content sizing, close button positioned absolute for alignment without affecting the grid's first row height (title)
    <div style={{ gridColumn: 'close / close', gridRow: 'header / content', width: 20 }}>
      <Toast.Close ref={ref} css={getCloseStyles(theme)} {...restProps} asChild={true}>
        <Button
          componentId={
            componentId ? componentId : 'codegen_design-system_src_design-system_notification_notification.tsx_224'
          }
          analyticsEvents={analyticsEvents}
          icon={<CloseIcon />}
          aria-label={closeLabel ?? restProps['aria-label'] ?? 'Close notification'}
        />
      </Toast.Close>
    </div>
  );
});

export interface NotificationProviderProps extends Toast.ToastProviderProps {}

export const Provider = ({ children, ...props }: NotificationProviderProps) => {
  return <Toast.Provider {...props}>{children}</Toast.Provider>;
};

export interface NotificationViewportProps extends Toast.ToastViewportProps {}

export const Viewport = (props: NotificationViewportProps): JSX.Element => {
  const { theme } = useDesignSystemTheme();

  return <Toast.Viewport className={DU_BOIS_ENABLE_ANIMATION_CLASSNAME} style={getViewportStyle(theme)} {...props} />;
};
