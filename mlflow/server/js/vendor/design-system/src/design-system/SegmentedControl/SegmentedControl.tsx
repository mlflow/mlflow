import type { CSSObject } from '@emotion/react';
import { css } from '@emotion/react';
import type { RadioGroupProps, RadioChangeEvent } from 'antd';
import { Radio as AntDRadio } from 'antd';
import type { RadioButtonProps } from 'antd/lib/radio/radioButton';
import React, {
  createContext,
  forwardRef,
  useContext,
  useCallback,
  useRef,
  useImperativeHandle,
  useEffect,
  useMemo,
} from 'react';

import type { Theme } from '../../theme';
import type { ButtonSize } from '../Button';
import {
  DesignSystemEventProviderAnalyticsEventTypes,
  DesignSystemEventProviderComponentTypes,
  useDesignSystemEventComponentCallbacks,
} from '../DesignSystemEventProvider/DesignSystemEventProvider';
import { DesignSystemAntDConfigProvider, getAnimationCss } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks/useDesignSystemTheme';
import type { AnalyticsEventValueChangeNoPiiFlagProps, DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
import { useDesignSystemSafexFlags } from '../utils';
import { importantify } from '../utils/css-utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
import { safex } from '../utils/safex';

// TODO(GP): Add this to common spacing vars; I didn't want to make a decision on the value right now,
// so copied it from `Button`.
const SMALL_BUTTON_HEIGHT = 24;

function getSegmentedControlGroupEmotionStyles(clsPrefix: string, spaced: boolean = false, truncateButtons: boolean) {
  const classGroup = `.${clsPrefix}-radio-group`;
  const classSmallGroup = `.${clsPrefix}-radio-group-small`;
  const classButtonWrapper = `.${clsPrefix}-radio-button-wrapper`;

  const styles: CSSObject = {
    ...(truncateButtons && {
      display: 'flex',
      maxWidth: '100%',
    }),
    [`&${classGroup}`]: spaced ? { display: 'flex', gap: 8, flexWrap: 'wrap' } : {},

    [`&${classSmallGroup} ${classButtonWrapper}`]: {
      padding: '0 12px',
    },
  };

  const importantStyles = importantify(styles);

  return css(importantStyles);
}

function getSegmentedControlButtonEmotionStyles(
  clsPrefix: string,
  theme: Theme,
  size: ButtonSize,
  spaced: boolean = false,
  truncateButtons: boolean,
  useNewShadows: boolean,
) {
  const classWrapperChecked = `.${clsPrefix}-radio-button-wrapper-checked`;
  const classWrapper = `.${clsPrefix}-radio-button-wrapper`;
  const classWrapperDisabled = `.${clsPrefix}-radio-button-wrapper-disabled`;
  const classButton = `.${clsPrefix}-radio-button`;

  // Note: Ant radio button uses a 1px-wide `before` pseudo-element to recreate the left border of the button.
  // This is because the actual left border is disabled to avoid a double-border effect with the adjacent button's
  // right border.
  // We must override the background colour of this pseudo-border to be the same as the real border above.

  const styles: CSSObject = {
    backgroundColor: theme.colors.actionDefaultBackgroundDefault,
    borderColor: theme.colors.actionDefaultBorderDefault,
    color: theme.colors.actionDefaultTextDefault,
    ...(useNewShadows && {
      boxShadow: theme.shadows.xs,
    }),

    // This handles the left border of the button when they're adjacent
    '::before': {
      display: spaced ? 'none' : 'block',
      backgroundColor: theme.colors.actionDefaultBorderDefault,
    },

    '&:hover': {
      backgroundColor: theme.colors.actionDefaultBackgroundHover,
      borderColor: theme.colors.actionDefaultBorderHover,
      color: theme.colors.actionDefaultTextHover,

      '::before': {
        backgroundColor: theme.colors.actionDefaultBorderHover,
      },

      // Also target the same pseudo-element on the next sibling, because this is used to create the right border
      [`& + ${classWrapper}::before`]: {
        backgroundColor: theme.colors.actionDefaultBorderPress,
      },
    },

    '&:active': {
      backgroundColor: theme.colors.actionTertiaryBackgroundPress,
      borderColor: theme.colors.actionDefaultBorderPress,
      color: theme.colors.actionTertiaryTextPress,
    },

    [`&${classWrapperChecked}`]: {
      backgroundColor: theme.colors.actionTertiaryBackgroundPress,
      borderColor: theme.colors.actionDefaultBorderPress,
      color: theme.colors.actionTertiaryTextPress,
      ...(!useNewShadows && {
        boxShadow: 'none',
      }),

      '::before': {
        backgroundColor: theme.colors.actionDefaultBorderPress,
      },

      [`& + ${classWrapper}::before`]: {
        backgroundColor: theme.colors.actionDefaultBorderPress,
      },
    },

    [`&${classWrapperChecked}:focus-within`]: {
      '::before': {
        width: 0,
      },
    },

    [`&${classWrapper}`]: {
      padding: size === 'middle' ? '0 16px' : '0 8px',
      display: 'inline-flex',
      verticalAlign: 'middle',
      ...(truncateButtons && {
        flexShrink: 1,
        textOverflow: 'ellipsis',
        whiteSpace: 'nowrap',
        minWidth: size === 'small' ? 58 : 68, // Don't allow the button to shrink and truncate below 3 characters
      }),

      ...(spaced
        ? {
            borderWidth: 1,
            borderRadius: theme.general.borderRadiusBase,
          }
        : {}),

      '&:focus-within': {
        outlineStyle: 'solid',
        outlineWidth: '2px',
        outlineOffset: '-2px',
        outlineColor: theme.colors.actionDefaultBorderFocus,
      },

      ...(truncateButtons && {
        'span:last-of-type': {
          textOverflow: 'ellipsis',
          overflow: 'hidden',
          whiteSpace: 'nowrap',
        },
      }),
    },

    [`&${classWrapper}, ${classButton}`]: {
      height: size === 'middle' ? theme.general.heightSm : SMALL_BUTTON_HEIGHT,
      lineHeight: theme.typography.lineHeightBase,
      alignItems: 'center',
    },

    [`&${classWrapperDisabled}, &${classWrapperDisabled} + ${classWrapperDisabled}`]: {
      color: theme.colors.actionDisabledText,
      backgroundColor: 'transparent',
      borderColor: theme.colors.actionDisabledBorder,

      '&:hover': {
        color: theme.colors.actionDisabledText,
        borderColor: theme.colors.actionDisabledBorder,
        backgroundColor: 'transparent',
      },
      '&:active': {
        color: theme.colors.actionDisabledText,
        borderColor: theme.colors.actionDisabledBorder,
        backgroundColor: 'transparent',
      },

      '::before': {
        backgroundColor: theme.colors.actionDisabledBorder,
      },

      [`&${classWrapperChecked}`]: {
        borderColor: theme.colors.actionDefaultBorderPress,
        '::before': {
          backgroundColor: theme.colors.actionDefaultBorderPress,
        },
      },
      [`&${classWrapperChecked} + ${classWrapper}`]: {
        '::before': {
          backgroundColor: theme.colors.actionDefaultBorderPress,
        },
      },
    },

    ...getAnimationCss(theme.options.enableAnimation),
  };

  const importantStyles = importantify(styles);

  return css(importantStyles);
}

const SegmentedControlGroupContext = createContext<{
  size: ButtonSize;
  spaced?: boolean;
}>({
  size: 'middle',
  spaced: false,
});
export interface SegmentedControlButtonProps
  extends Omit<RadioButtonProps, 'optionType' | 'buttonStyle' | 'prefixCls' | 'skipGroup'>,
    DangerouslySetAntdProps<RadioButtonProps>,
    HTMLDataAttributes {}

export const SegmentedControlButton = forwardRef<HTMLButtonElement, SegmentedControlButtonProps>(
  function SegmentedControlButton(
    { dangerouslySetAntdProps, ...props }: SegmentedControlButtonProps,
    ref,
  ): JSX.Element {
    const { classNamePrefix, theme } = useDesignSystemTheme();
    const { size, spaced } = useContext(SegmentedControlGroupContext);
    const truncateButtons = safex('databricks.fe.designsystem.truncateSegmentedControlText', false);
    const { useNewShadows } = useDesignSystemSafexFlags();
    const buttonRef = useRef<HTMLButtonElement>(null);
    useImperativeHandle(ref, () => buttonRef.current as HTMLButtonElement);

    const getLabelFromChildren = useCallback((): string => {
      let label = '';

      React.Children.map(props.children, (child) => {
        if (typeof child === 'string') {
          label += child;
        }
      });

      return label;
    }, [props.children]);

    useEffect(() => {
      if (buttonRef.current) {
        // Using `as any` because Antd uses a `Checkbox` type that's not exported
        const labelParent = (buttonRef.current as any).input.closest('label');

        if (labelParent) {
          labelParent.setAttribute('title', getLabelFromChildren());
        }
      }
    }, [buttonRef, getLabelFromChildren]);

    return (
      <DesignSystemAntDConfigProvider>
        <AntDRadio.Button
          css={getSegmentedControlButtonEmotionStyles(
            classNamePrefix,
            theme,
            size,
            spaced,
            truncateButtons,
            useNewShadows,
          )}
          {...props}
          {...dangerouslySetAntdProps}
          ref={buttonRef}
        />
      </DesignSystemAntDConfigProvider>
    );
  },
);

export interface SegmentedControlGroupProps
  extends Omit<RadioGroupProps, 'size'>,
    DangerouslySetAntdProps<RadioGroupProps>,
    HTMLDataAttributes,
    AnalyticsEventValueChangeNoPiiFlagProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange> {
  size?: ButtonSize;
  spaced?: boolean;
  name: string;
}

export const SegmentedControlGroup = forwardRef<HTMLDivElement, SegmentedControlGroupProps>(
  function SegmentedControlGroup(
    {
      dangerouslySetAntdProps,
      size = 'middle',
      spaced = false,
      onChange,
      componentId,
      analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange],
      valueHasNoPii,
      ...props
    }: SegmentedControlGroupProps,
    ref,
  ): JSX.Element {
    const { classNamePrefix } = useDesignSystemTheme();
    const truncateButtons = safex('databricks.fe.designsystem.truncateSegmentedControlText', false);
    const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
    const eventContext = useDesignSystemEventComponentCallbacks({
      componentType: DesignSystemEventProviderComponentTypes.SegmentedControlGroup,
      componentId,
      analyticsEvents: memoizedAnalyticsEvents,
      valueHasNoPii,
    });

    const onChangeWrapper = useCallback(
      (e: RadioChangeEvent) => {
        eventContext.onValueChange(e.target.value);

        onChange?.(e);
      },
      [eventContext, onChange],
    );

    return (
      <DesignSystemAntDConfigProvider>
        <SegmentedControlGroupContext.Provider value={{ size, spaced }}>
          <AntDRadio.Group
            {...addDebugOutlineIfEnabled()}
            {...props}
            css={getSegmentedControlGroupEmotionStyles(classNamePrefix, spaced, truncateButtons)}
            onChange={onChangeWrapper}
            {...dangerouslySetAntdProps}
            ref={ref}
            {...eventContext.dataComponentProps}
          />
        </SegmentedControlGroupContext.Provider>
      </DesignSystemAntDConfigProvider>
    );
  },
);
