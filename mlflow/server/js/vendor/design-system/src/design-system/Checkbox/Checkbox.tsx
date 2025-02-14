import type { CSSObject } from '@emotion/react';
import { css } from '@emotion/react';
import { Checkbox as AntDCheckbox } from 'antd';
import type {
  CheckboxProps as AntDCheckboxProps,
  CheckboxGroupProps as AntDCheckboxGroupProps,
  CheckboxChangeEvent,
} from 'antd/lib/checkbox';
import type { CheckboxValueType as AntDCheckboxValueType } from 'antd/lib/checkbox/Group';
import classnames from 'classnames';
import { forwardRef, useMemo } from 'react';

import type { Theme } from '../../theme';
import {
  DesignSystemEventProviderAnalyticsEventTypes,
  DesignSystemEventProviderComponentTypes,
  useDesignSystemEventComponentCallbacks,
} from '../DesignSystemEventProvider';
import { DesignSystemAntDConfigProvider, getAnimationCss, RestoreAntDDefaultClsPrefix } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks/useDesignSystemTheme';
import type { AnalyticsEventProps, DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
import { useDesignSystemSafexFlags } from '../utils';
import { importantify } from '../utils/css-utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';

export type CheckboxValueType = AntDCheckboxValueType;

function getCheckboxEmotionStyles(
  clsPrefix: string,
  theme: Theme,
  isHorizontal = false,
  useNewShadows: boolean,
  useNewFormUISpacing: boolean,
): CSSObject {
  const classInput = `.${clsPrefix}-input`;
  const classInner = `.${clsPrefix}-inner`;
  const classIndeterminate = `.${clsPrefix}-indeterminate`;
  const classChecked = `.${clsPrefix}-checked`;
  const classDisabled = `.${clsPrefix}-disabled`;
  const classDisabledWrapper = `.${clsPrefix}-wrapper-disabled`;
  const classContainer = `.${clsPrefix}-group`;
  const classWrapper = `.${clsPrefix}-wrapper`;

  const defaultSelector = `${classInput} + ${classInner}`;
  const hoverSelector = `${classInput}:hover + ${classInner}`;
  const pressSelector = `${classInput}:active + ${classInner}`;

  const cleanClsPrefix = `.${clsPrefix.replace('-checkbox', '')}`;

  const styles: CSSObject = {
    [`.${clsPrefix}`]: {
      top: 'unset',
      lineHeight: theme.typography.lineHeightBase,
    },

    [`&${classWrapper}, ${classWrapper}`]: {
      alignItems: 'center',
      lineHeight: theme.typography.lineHeightBase,
    },

    // Top level styles are for the unchecked state
    [classInner]: {
      borderColor: theme.colors.actionDefaultBorderDefault,
    },

    // Style wrapper span added by Antd
    [`&> span:not(.${clsPrefix})`]: {
      display: 'inline-flex',
      alignItems: 'center',
    },

    // Layout styling
    [`&${classContainer}`]: {
      display: 'flex',
      flexDirection: 'column',
      rowGap: theme.spacing.sm,
      columnGap: 0,

      ...(useNewFormUISpacing && {
        [`& + ${cleanClsPrefix}-form-message`]: {
          marginTop: theme.spacing.sm,
        },
      }),
    },

    ...(useNewFormUISpacing && {
      [`${cleanClsPrefix}-hint + &${classContainer}`]: {
        marginTop: theme.spacing.sm,
      },
    }),

    ...(isHorizontal && {
      [`&${classContainer}`]: {
        display: 'flex',
        flexDirection: 'row',
        columnGap: theme.spacing.sm,
        rowGap: 0,

        [`& > ${classContainer}-item`]: {
          marginRight: 0,
        },
      },
    }),

    // Keyboard focus
    [`${classInput}:focus-visible + ${classInner}`]: {
      outlineWidth: '2px',
      outlineColor: theme.colors.actionDefaultBorderFocus,
      outlineOffset: '4px',
      outlineStyle: 'solid',
    },

    // Hover
    [hoverSelector]: {
      backgroundColor: theme.colors.actionDefaultBackgroundHover,
      borderColor: theme.colors.actionPrimaryBackgroundHover,
    },

    // Mouse pressed
    [pressSelector]: {
      backgroundColor: theme.colors.actionDefaultBackgroundPress,
      borderColor: theme.colors.actionPrimaryBackgroundPress,
    },

    // Checked state
    [classChecked]: {
      ...(useNewShadows && {
        [classInner]: {
          boxShadow: theme.shadows.xs,
        },
      }),

      '&::after': {
        border: 'none',
      },

      [defaultSelector]: {
        backgroundColor: theme.colors.actionPrimaryBackgroundDefault,
        borderColor: 'transparent',
      },

      // Checked hover
      [hoverSelector]: {
        backgroundColor: theme.colors.actionPrimaryBackgroundHover,
        borderColor: 'transparent',
      },

      // Checked and mouse pressed
      [pressSelector]: {
        backgroundColor: theme.colors.actionPrimaryBackgroundPress,
        borderColor: 'transparent',
      },
    },

    // Indeterminate
    [classIndeterminate]: {
      [classInner]: {
        ...(useNewShadows && {
          boxShadow: theme.shadows.xs,
        }),
        backgroundColor: theme.colors.actionPrimaryBackgroundDefault,
        borderColor: theme.colors.actionPrimaryBackgroundDefault,

        // The after pseudo-element is used for the check image itself
        '&:after': {
          backgroundColor: theme.colors.white,
          height: '3px',
          width: '8px',
          borderRadius: '4px',
        },
      },

      // Indeterminate hover
      [hoverSelector]: {
        backgroundColor: theme.colors.actionPrimaryBackgroundHover,
        borderColor: 'transparent',
      },

      // Indeterminate and mouse pressed
      [pressSelector]: {
        backgroundColor: theme.colors.actionPrimaryBackgroundPress,
      },
    },

    // Disabled state
    [`&${classDisabledWrapper}`]: {
      [classDisabled]: {
        // Disabled Checked
        [`&${classChecked}`]: {
          [classInner]: {
            backgroundColor: theme.colors.actionDisabledBackground,
            borderColor: theme.colors.actionDisabledBorder,

            '&:after': {
              borderColor: theme.colors.actionDisabledText,
            },
          },

          // Disabled checked hover
          [hoverSelector]: {
            backgroundColor: theme.colors.actionDisabledBackground,
            borderColor: theme.colors.actionDisabledBorder,
          },
        },

        // Disabled indeterminate
        [`&${classIndeterminate}`]: {
          [classInner]: {
            backgroundColor: theme.colors.actionDisabledBackground,
            borderColor: theme.colors.actionDisabledBorder,

            '&:after': {
              borderColor: theme.colors.actionDisabledText,
              backgroundColor: theme.colors.actionDisabledText,
            },
          },

          // Disabled indeterminate hover
          [hoverSelector]: {
            backgroundColor: theme.colors.actionDisabledBackground,
            borderColor: theme.colors.actionDisabledBorder,
          },
        },

        // Disabled unchecked
        [classInner]: {
          backgroundColor: theme.colors.actionDisabledBackground,
          borderColor: theme.colors.actionDisabledBorder,

          // The after pseudo-element is used for the check image itself
          '&:after': {
            borderColor: 'transparent',
          },
        },

        // Disabled hover
        [hoverSelector]: {
          backgroundColor: theme.colors.actionDisabledBackground,
          borderColor: theme.colors.actionDisabledBorder,
        },

        '& + span': {
          color: theme.colors.actionDisabledText,
        },
      },
    },

    // Animation
    ...getAnimationCss(theme.options.enableAnimation),
  };

  return styles;
}

export const getWrapperStyle = ({
  clsPrefix,
  theme,
  wrapperStyle = {},
  useNewFormUISpacing,
}: {
  clsPrefix: string;
  theme: Theme;
  wrapperStyle?: React.CSSProperties;
  useNewStyles?: boolean;
  useNewFormUISpacing?: boolean;
}) => {
  const extraSelector = useNewFormUISpacing ? `, && + .${clsPrefix}-hint + .${clsPrefix}-form-message` : '';

  const styles = {
    height: theme.typography.lineHeightBase,
    lineHeight: theme.typography.lineHeightBase,

    [`&& + .${clsPrefix}-hint, && + .${clsPrefix}-form-message${extraSelector}`]: {
      paddingLeft: theme.spacing.lg,
      marginTop: 0,
    },

    ...wrapperStyle,
  };

  return css(styles);
};

export interface CheckboxProps
  extends DangerouslySetAntdProps<AntDCheckboxProps>,
    Omit<React.InputHTMLAttributes<HTMLInputElement>, 'onChange' | 'checked'>,
    HTMLDataAttributes,
    AnalyticsEventProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange> {
  isChecked?: boolean | null;
  onChange?: (isChecked: boolean, event: CheckboxChangeEvent) => void;
  children?: React.ReactNode;
  isDisabled?: boolean;
  /**
   * Used to set styling for the div wrapping the checkbox element
   */
  wrapperStyle?: React.CSSProperties;
  /**
   * Used to style the checkbox element itself
   */
  style?: React.CSSProperties;
}

export interface CheckboxGroupProps
  extends Omit<AntDCheckboxGroupProps, 'prefixCls'>,
    Omit<React.InputHTMLAttributes<HTMLInputElement>, 'defaultValue' | 'onChange' | 'value'> {
  layout?: 'vertical' | 'horizontal';
}

const DuboisCheckbox = forwardRef<HTMLInputElement, CheckboxProps>(function Checkbox(
  {
    isChecked,
    onChange,
    children,
    isDisabled = false,
    style,
    wrapperStyle,
    dangerouslySetAntdProps,
    className,
    componentId,
    analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange],
    ...restProps
  }: CheckboxProps,
  ref,
): JSX.Element {
  const { theme, classNamePrefix, getPrefixedClassName } = useDesignSystemTheme();
  const { useNewShadows, useNewFormUISpacing } = useDesignSystemSafexFlags();
  const clsPrefix = getPrefixedClassName('checkbox');
  const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.Checkbox,
    componentId,
    analyticsEvents: memoizedAnalyticsEvents,
    valueHasNoPii: true,
  });

  const onChangeHandler = (event: CheckboxChangeEvent) => {
    eventContext.onValueChange(event.target.checked);
    onChange?.(event.target.checked, event);
  };

  return (
    <DesignSystemAntDConfigProvider>
      {/* Antd checkboxes are inline, so we add this div to make ours block-level */}
      <div
        {...addDebugOutlineIfEnabled()}
        className={classnames(className, `${clsPrefix}-container`)}
        css={getWrapperStyle({
          clsPrefix: classNamePrefix,
          theme,
          wrapperStyle,
          useNewFormUISpacing,
        })}
      >
        <AntDCheckbox
          checked={isChecked === null ? undefined : isChecked}
          ref={ref}
          onChange={onChangeHandler}
          disabled={isDisabled}
          indeterminate={isChecked === null}
          // Individual checkboxes don't depend on isHorizontal flag, orientation and spacing is handled by end users
          css={css(importantify(getCheckboxEmotionStyles(clsPrefix, theme, false, useNewShadows, useNewFormUISpacing)))}
          style={style}
          aria-checked={isChecked === null ? 'mixed' : isChecked}
          {...restProps}
          {...dangerouslySetAntdProps}
          {...eventContext.dataComponentProps}
        >
          <RestoreAntDDefaultClsPrefix>{children}</RestoreAntDDefaultClsPrefix>
        </AntDCheckbox>
      </div>
    </DesignSystemAntDConfigProvider>
  );
});

const CheckboxGroup = forwardRef<HTMLInputElement, CheckboxGroupProps>(function CheckboxGroup(
  { children, layout = 'vertical', ...props }: CheckboxGroupProps,
  ref,
): JSX.Element {
  const { theme, getPrefixedClassName } = useDesignSystemTheme();
  const clsPrefix = getPrefixedClassName('checkbox');
  const { useNewShadows, useNewFormUISpacing } = useDesignSystemSafexFlags();

  return (
    <DesignSystemAntDConfigProvider>
      <AntDCheckbox.Group
        {...addDebugOutlineIfEnabled()}
        ref={ref}
        {...props}
        css={getCheckboxEmotionStyles(clsPrefix, theme, layout === 'horizontal', useNewShadows, useNewFormUISpacing)}
      >
        <RestoreAntDDefaultClsPrefix>{children}</RestoreAntDDefaultClsPrefix>
      </AntDCheckbox.Group>
    </DesignSystemAntDConfigProvider>
  );
});

const CheckboxNamespace = /* #__PURE__ */ Object.assign(DuboisCheckbox, { Group: CheckboxGroup });

export const Checkbox = CheckboxNamespace;

// TODO: I'm doing this to support storybook's docgen;
// We should remove this once we have a better storybook integration,
// since these will be exposed in the library's exports.
export const __INTERNAL_DO_NOT_USE__Group = CheckboxGroup;
