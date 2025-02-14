import type { CSSObject, SerializedStyles } from '@emotion/react';
import { css } from '@emotion/react';
import type { WithConditionalCSSProp } from '@emotion/react/types/jsx-namespace';
import type { SwitchProps as AntDSwitchProps } from 'antd';
import { Switch as AntDSwitch } from 'antd';
import { useEffect, useMemo, useState } from 'react';

import type { Theme } from '../../theme';
import {
  useDesignSystemEventComponentCallbacks,
  DesignSystemEventProviderAnalyticsEventTypes,
  DesignSystemEventProviderComponentTypes,
} from '../DesignSystemEventProvider';
import { DesignSystemAntDConfigProvider, getAnimationCss } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks/useDesignSystemTheme';
import type { LabelProps } from '../Label/Label';
import { Label } from '../Label/Label';
import type { AnalyticsEventProps, DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
import { useDesignSystemSafexFlags } from '../utils';
import { importantify } from '../utils/css-utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
import { useUniqueId } from '../utils/useUniqueId';

export interface SwitchProps
  extends Pick<
      AntDSwitchProps,
      | 'autoFocus'
      | 'checked'
      | 'checkedChildren'
      | 'className'
      | 'defaultChecked'
      | 'disabled'
      | 'unCheckedChildren'
      | 'onChange'
      | 'onClick'
    >,
    HTMLDataAttributes,
    DangerouslySetAntdProps<AntDSwitchProps>,
    AnalyticsEventProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange> {
  id?: string;
  /**
   * Label for the Switch, provided as prop for styling purposes
   */
  label?: string;
  labelProps?: LabelProps & WithConditionalCSSProp<LabelProps>;
  activeLabel?: React.ReactNode;
  inactiveLabel?: React.ReactNode;
  disabledLabel?: React.ReactNode;
}

const getSwitchWithLabelStyles = ({
  clsPrefix,
  theme,
  disabled,
  useNewShadows,
}: {
  clsPrefix: string;
  theme: Theme;
  disabled?: boolean;
  useNewShadows: boolean;
}): SerializedStyles => {
  // Default value
  const SWITCH_WIDTH = 28;

  const styles: CSSObject = {
    display: 'flex',
    alignItems: 'center',

    ...(disabled && {
      '&&, label': {
        color: theme.colors.actionDisabledText,
      },
    }),

    ...(useNewShadows && {
      [`&.${clsPrefix}-switch, &.${clsPrefix}-switch-checked`]: {
        [`.${clsPrefix}-switch-handle`]: {
          top: -1,
        },

        [`.${clsPrefix}-switch-handle, .${clsPrefix}-switch-handle:before`]: {
          width: 16,
          height: 16,
          borderRadius: 99999,
        },
      },
    }),

    // Switch is Off
    [`&.${clsPrefix}-switch`]: {
      backgroundColor: theme.colors.backgroundPrimary,
      border: `1px solid ${theme.colors.actionDefaultBorderDefault}`,

      [`.${clsPrefix}-switch-handle:before`]: {
        ...(useNewShadows
          ? {
              border: `1px solid ${theme.colors.actionDefaultBorderDefault}`,
              boxShadow: theme.shadows.xs,
              left: -1,
            }
          : {
              boxShadow: `0px 0px 0px 1px ${theme.colors.actionDefaultBorderDefault}`,
            }),
        transition: 'none',
      },

      [`&:hover:not(.${clsPrefix}-switch-disabled)`]: {
        backgroundColor: theme.colors.actionDefaultBackgroundHover,
        border: `1px solid ${theme.colors.actionPrimaryBackgroundHover}`,

        [`.${clsPrefix}-switch-handle:before`]: {
          ...(useNewShadows
            ? {
                boxShadow: theme.shadows.xs,
                border: `1px solid ${theme.colors.actionPrimaryBackgroundHover}`,
              }
            : {
                boxShadow: `0px 0px 0px 1px ${theme.colors.actionPrimaryBackgroundHover}`,
              }),
        },
      },

      [`&:active:not(.${clsPrefix}-switch-disabled)`]: {
        backgroundColor: theme.colors.actionDefaultBackgroundPress,
        border: `1px solid ${theme.colors.actionPrimaryBackgroundPress}`,

        [`.${clsPrefix}-switch-handle:before`]: {
          ...(useNewShadows
            ? {
                boxShadow: 'none',
                border: `1px solid ${theme.colors.actionPrimaryBackgroundPress}`,
              }
            : {
                boxShadow: `0px 0px 0px 1px ${theme.colors.actionPrimaryBackgroundHover}`,
              }),
        },
      },

      [`&.${clsPrefix}-switch-disabled`]: {
        backgroundColor: theme.colors.actionDisabledBackground,
        border: `1px solid ${theme.colors.actionDisabledBorder}`,

        [`.${clsPrefix}-switch-handle:before`]: {
          ...(useNewShadows
            ? {
                boxShadow: 'none',
                border: `1px solid ${theme.colors.actionDisabledBorder}`,
              }
            : {
                boxShadow: `0px 0px 0px 1px ${theme.colors.actionDisabledBorder}`,
              }),
        },
      },

      [`&:focus-visible`]: {
        border: `1px solid ${theme.colors.actionPrimaryBackgroundDefault}`,
        boxShadow: 'none',
        outlineStyle: 'solid',
        outlineWidth: '1px',
        outlineColor: theme.colors.actionDefaultBorderFocus,

        [`.${clsPrefix}-switch-handle:before`]: {
          ...(useNewShadows
            ? {
                boxShadow: theme.shadows.xs,
                border: `1px solid ${theme.colors.actionPrimaryBackgroundDefault}`,
              }
            : {
                boxShadow: `0px 0px 0px 1px ${theme.colors.actionPrimaryBackgroundDefault}`,
              }),
        },
      },

      [`&:focus`]: {
        boxShadow: 'none',
      },
    },

    // Switch is On
    [`&.${clsPrefix}-switch-checked`]: {
      backgroundColor: theme.colors.actionPrimaryBackgroundDefault,
      border: `1px solid ${theme.colors.actionPrimaryBackgroundDefault}`,

      [`&:hover:not(.${clsPrefix}-switch-disabled)`]: {
        backgroundColor: theme.colors.actionPrimaryBackgroundHover,
        ...(useNewShadows
          ? {
              border: `1px solid ${theme.colors.actionPrimaryBackgroundHover}`,

              [`.${clsPrefix}-switch-handle:before`]: {
                border: `1px solid ${theme.colors.actionPrimaryBackgroundHover}`,
              },
            }
          : {
              border: `1px solid ${theme.colors.actionPrimaryBackgroundDefault}`,
            }),
      },

      [`&:active:not(.${clsPrefix}-switch-disabled)`]: {
        backgroundColor: theme.colors.actionPrimaryBackgroundPress,
        ...(useNewShadows && {
          border: `1px solid ${theme.colors.actionPrimaryBackgroundPress}`,

          [`.${clsPrefix}-switch-handle:before`]: {
            border: `1px solid ${theme.colors.actionPrimaryBackgroundPress}`,
          },
        }),
      },

      [`.${clsPrefix}-switch-handle:before`]: {
        ...(useNewShadows
          ? {
              boxShadow: theme.shadows.xs,
              border: `1px solid ${theme.colors.actionPrimaryBackgroundDefault}`,
              right: -1,
            }
          : {
              boxShadow: `0px 0px 0px 1px ${theme.colors.actionPrimaryBackgroundDefault}`,
            }),
      },

      [`&.${clsPrefix}-switch-disabled`]: {
        backgroundColor: theme.colors.actionDisabledText,
        border: `1px solid ${theme.colors.actionDisabledText}`,

        [`.${clsPrefix}-switch-handle:before`]: {
          ...(useNewShadows
            ? {
                boxShadow: 'none',
                border: `1px solid ${theme.colors.actionDisabledText}`,
              }
            : {
                boxShadow: `0px 0px 0px 1px ${theme.colors.actionDisabledText}`,
              }),
        },
      },

      [`&:focus-visible`]: {
        outlineOffset: '1px',
      },
    },

    [`.${clsPrefix}-switch-handle:before`]: {
      backgroundColor: theme.colors.backgroundPrimary,
    },

    [`&& + .${clsPrefix}-hint, && + .${clsPrefix}-form-message`]: {
      paddingLeft: theme.spacing.sm + SWITCH_WIDTH,
    },

    [`&& + .${clsPrefix}-form-message`]: {
      marginTop: 0,
    },

    [`.${clsPrefix}-click-animating-node`]: {
      animation: 'none',
    },

    opacity: 1,
  };
  const importantStyles = importantify(styles);

  return css(importantStyles);
};

export const Switch: React.FC<SwitchProps> = ({
  dangerouslySetAntdProps,
  label,
  labelProps,
  activeLabel,
  inactiveLabel,
  disabledLabel,
  componentId,
  analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange],
  onChange,
  ...props
}) => {
  const { theme, classNamePrefix } = useDesignSystemTheme();
  const duboisId = useUniqueId('dubois-switch');
  const { useNewShadows } = useDesignSystemSafexFlags();
  const uniqueId = props.id ?? duboisId;
  const [isChecked, setIsChecked] = useState(props.checked || props.defaultChecked);
  const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.Switch,
    componentId,
    analyticsEvents: memoizedAnalyticsEvents,
    valueHasNoPii: true,
  });

  const handleToggle = (newState: boolean, event: MouseEvent) => {
    eventContext.onValueChange(newState);
    if (onChange) {
      onChange(newState, event);
    } else {
      setIsChecked(newState);
    }
  };

  const onChangeHandler = (newState: boolean, event: MouseEvent) => {
    eventContext.onValueChange(newState);
    onChange?.(newState, event);
  };

  useEffect(() => {
    setIsChecked(props.checked);
  }, [props.checked]);

  const hasNewLabels = activeLabel && inactiveLabel && disabledLabel;
  const stateMessage = isChecked ? activeLabel : inactiveLabel;

  // AntDSwitch's interface does not include `id` even though it passes it through and works as expected
  // We are using this to bypass that check
  const idPropObj = {
    id: uniqueId,
  };

  const switchComponent = (
    <AntDSwitch
      {...addDebugOutlineIfEnabled()}
      {...props}
      {...dangerouslySetAntdProps}
      onChange={handleToggle}
      {...idPropObj}
      css={{
        ...css(getAnimationCss(theme.options.enableAnimation)),
        ...getSwitchWithLabelStyles({ clsPrefix: classNamePrefix, theme, disabled: props.disabled, useNewShadows }),
      }}
    />
  );

  const labelComponent = (
    <Label
      inline={true}
      {...labelProps}
      htmlFor={uniqueId}
      style={{ ...(hasNewLabels && { marginRight: theme.spacing.sm }) }}
    >
      {label}
    </Label>
  );

  return label ? (
    <DesignSystemAntDConfigProvider>
      {/*We need to offer the component with embedded Label in order to control the label alignment to Switch*/}
      {/*and also control the padding of Hints and FormMessage relative to the Switch's label*/}
      <div
        {...addDebugOutlineIfEnabled()}
        css={getSwitchWithLabelStyles({ clsPrefix: classNamePrefix, theme, disabled: props.disabled, useNewShadows })}
        {...eventContext.dataComponentProps}
      >
        {hasNewLabels ? (
          <>
            {labelComponent}
            <span style={{ marginLeft: 'auto', marginRight: theme.spacing.sm }}>
              {`${stateMessage}${props.disabled ? ` (${disabledLabel})` : ''}`}
            </span>
            {switchComponent}
          </>
        ) : (
          <>
            {switchComponent}
            {labelComponent}
          </>
        )}
      </div>
    </DesignSystemAntDConfigProvider>
  ) : (
    <DesignSystemAntDConfigProvider>
      <AntDSwitch
        onChange={onChangeHandler}
        {...addDebugOutlineIfEnabled()}
        {...props}
        {...dangerouslySetAntdProps}
        {...idPropObj}
        css={{
          ...css(getAnimationCss(theme.options.enableAnimation)),
          ...getSwitchWithLabelStyles({ clsPrefix: classNamePrefix, theme, disabled: props.disabled, useNewShadows }),
        }}
        {...eventContext.dataComponentProps}
      />
    </DesignSystemAntDConfigProvider>
  );
};
