import type { CSSObject } from '@emotion/react';
import { css } from '@emotion/react';
import type { AlertProps as AntDAlertProps } from 'antd';
import { Alert as AntDAlert } from 'antd';
import cx from 'classnames';
import { useEffect, useMemo, useRef } from 'react';

import type { Theme } from '../../theme';
import {
  DesignSystemEventProviderAnalyticsEventTypes,
  DesignSystemEventProviderComponentTypes,
  useDesignSystemEventComponentCallbacks,
  DesignSystemEventProviderComponentSubTypeMap,
} from '../DesignSystemEventProvider';
import { DesignSystemAntDConfigProvider, getAnimationCss } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks';
import { CloseIcon } from '../Icon';
import { SeverityIcon } from '../Icon/iconMap';
import type { AnalyticsEventProps, DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
import { useNotifyOnFirstView } from '../utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';

export type AlertType = NonNullable<Exclude<AntDAlertProps['type'], 'success'>>;

export interface AlertProps
  extends Omit<AntDAlertProps, 'closeText' | 'showIcon' | 'type' | 'icon'>,
    HTMLDataAttributes,
    DangerouslySetAntdProps<AntDAlertProps>,
    AnalyticsEventProps<DesignSystemEventProviderAnalyticsEventTypes.OnView> {
  type: AlertType;
  closeIconLabel?: string;
  /**
   * @deprecated Use CEP-governed banners instead. go/banner/create go/getabanner go/banner
   */
  banner?: boolean;
}

export const Alert: React.FC<AlertProps> = ({
  componentId,
  analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnView],
  dangerouslySetAntdProps,
  closable = true,
  closeIconLabel = 'Close alert',
  onClose,
  ...props
}) => {
  const { theme, getPrefixedClassName } = useDesignSystemTheme();
  const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.Alert,
    componentId,
    componentSubType: DesignSystemEventProviderComponentSubTypeMap[props.type],
    analyticsEvents: memoizedAnalyticsEvents,
  });
  const closeButtonEventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.Button,
    componentId: componentId ? `${componentId}.close` : 'codegen_design_system_src_design_system_alert_alert.tsx_50',
    analyticsEvents: [DesignSystemEventProviderAnalyticsEventTypes.OnClick],
  });
  const { elementRef } = useNotifyOnFirstView<HTMLSpanElement>({ onView: eventContext.onView });
  const clsPrefix = getPrefixedClassName('alert');

  const mergedProps: AntDAlertProps & { type: AlertType } = {
    ...props,
    type: props.type || 'error',
    showIcon: true,
    closable,
  };

  const closeIconRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (closeIconRef.current) {
      closeIconRef.current.removeAttribute('aria-label');
      closeIconRef.current.closest('button')?.setAttribute('aria-label', closeIconLabel);
    }
  }, [mergedProps.closable, closeIconLabel, closeIconRef]);

  const onCloseWrapper: React.MouseEventHandler<HTMLButtonElement> = (e) => {
    closeButtonEventContext.onClick(e);
    onClose?.(e);
  };

  return (
    <DesignSystemAntDConfigProvider>
      <AntDAlert
        {...addDebugOutlineIfEnabled()}
        {...mergedProps}
        onClose={onCloseWrapper}
        className={cx(mergedProps.className)}
        css={getAlertEmotionStyles(clsPrefix, theme, mergedProps)}
        icon={<SeverityIcon severity={mergedProps.type} ref={elementRef} />}
        // Antd calls this prop `closeText` but we can use it to set any React element to replace the close icon.
        closeText={
          mergedProps.closable && (
            <CloseIcon ref={closeIconRef} aria-label={closeIconLabel} css={{ fontSize: theme.general.iconSize }} />
          )
        }
        // Always set a description for consistent styling (e.g. icon size)
        description={props.description || ' '}
        {...dangerouslySetAntdProps}
        {...eventContext.dataComponentProps}
      />
    </DesignSystemAntDConfigProvider>
  );
};

const getAlertEmotionStyles = (clsPrefix: string, theme: Theme, props: AntDAlertProps) => {
  const classCloseIcon = `.${clsPrefix}-close-icon`;
  const classCloseButton = `.${clsPrefix}-close-button`;
  const classCloseText = `.${clsPrefix}-close-text`;
  const classDescription = `.${clsPrefix}-description`;
  const classMessage = `.${clsPrefix}-message`;
  const classWithDescription = `.${clsPrefix}-with-description`;
  const classWithIcon = `.${clsPrefix}-icon`;

  const ALERT_ICON_HEIGHT = 16;
  const ALERT_ICON_FONT_SIZE = 16;

  const styles: CSSObject = {
    // General
    padding: theme.spacing.sm,

    [`${classMessage}, &${classWithDescription} ${classMessage}`]: {
      // TODO(giles): These three rules are all the same as the H3 styles. We can refactor them out into a shared object.
      fontSize: theme.typography.fontSizeBase,
      fontWeight: theme.typography.typographyBoldFontWeight,
      lineHeight: theme.typography.lineHeightBase,
    },

    [`${classDescription}`]: {
      lineHeight: theme.typography.lineHeightBase,
    },

    // Icons
    [classCloseButton]: {
      fontSize: ALERT_ICON_FONT_SIZE,
      marginRight: 12,
    },
    [classCloseIcon]: {
      '&:focus-visible': {
        outlineStyle: 'auto',
        outlineColor: theme.colors.actionDefaultBorderFocus,
      },
    },
    [`${classCloseIcon}, ${classCloseButton}`]: {
      lineHeight: theme.typography.lineHeightBase,
      height: ALERT_ICON_HEIGHT,
      width: ALERT_ICON_HEIGHT,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
    },

    [classWithIcon]: {
      fontSize: ALERT_ICON_FONT_SIZE,
      marginTop: 2,
    },
    [`${classCloseIcon}, ${classCloseButton}, ${classCloseText} > span`]: {
      lineHeight: theme.typography.lineHeightBase,
      height: ALERT_ICON_HEIGHT,
      width: ALERT_ICON_HEIGHT,
      fontSize: ALERT_ICON_FONT_SIZE,
      marginTop: 2,

      '& > span': {
        lineHeight: theme.typography.lineHeightBase,
      },
    },

    // No description
    ...(!props.description && {
      display: 'flex',
      alignItems: 'center',
      [classWithIcon]: {
        fontSize: ALERT_ICON_FONT_SIZE,
        marginTop: 0,
      },
      [classMessage]: {
        margin: 0,
      },
      [classDescription]: {
        display: 'none',
      },
      [classCloseIcon]: {
        alignSelf: 'baseline',
      },
    }),

    // Warning
    ...(props.type === 'warning' && {
      color: theme.colors.textValidationWarning,
      borderColor: theme.colors.yellow300,
    }),

    // Error
    ...(props.type === 'error' && {
      color: theme.colors.textValidationDanger,
      borderColor: theme.colors.red300,
    }),

    // Banner
    ...(props.banner && {
      borderStyle: 'solid',
      borderWidth: `${theme.general.borderWidth}px 0`,
    }),

    // After closed
    '&[data-show="false"]': {
      borderWidth: 0,
      padding: 0,
      width: 0,
      height: 0,
    },

    ...getAnimationCss(theme.options.enableAnimation),
  };

  return css(styles);
};
