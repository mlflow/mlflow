import type { CSSObject, SerializedStyles } from '@emotion/react';
import { css } from '@emotion/react';
import { DatePicker as AntDDatePicker } from 'antd';
import type {
  DatePickerProps as AntDDatePickerProps,
  MonthPickerProps as AntDMonthPickerProps,
  RangePickerProps as AntDRangePickerProps,
} from 'antd/lib/date-picker';
import React, { forwardRef, useEffect, useRef } from 'react';

import type { Theme } from '../../theme';
import { DesignSystemAntDConfigProvider } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks/useDesignSystemTheme';
import { addDebugOutlineIfEnabled } from '../utils/debug';

export type DatePickerRef = {
  focus: () => void;
  blur: () => void;
};

type DatePickerAccessibilityProps = {
  /**
   * ARIA live region attribute for the DatePicker component.
   * @default 'assertive'
   * It's recommended to use `assertive` as `polite` has been known to cause issues with screen readers with this specific component.
   */
  ariaLive?: 'assertive' | 'polite';
  wrapperDivProps?: React.HTMLAttributes<HTMLDivElement>;
};

export type DatePickerProps = AntDDatePickerProps & React.RefAttributes<DatePickerRef> & DatePickerAccessibilityProps;
export type RangePickerProps = AntDRangePickerProps & React.RefAttributes<DatePickerRef> & DatePickerAccessibilityProps;
export type TimePickerProps = AntDMonthPickerProps & React.RefAttributes<DatePickerRef> & DatePickerAccessibilityProps;
export type QuarterPickerProps = AntDMonthPickerProps &
  React.RefAttributes<DatePickerRef> &
  DatePickerAccessibilityProps;
export type WeekPickerProps = AntDMonthPickerProps & React.RefAttributes<DatePickerRef> & DatePickerAccessibilityProps;
export type MonthPickerProps = AntDMonthPickerProps & React.RefAttributes<DatePickerRef> & DatePickerAccessibilityProps;
export type YearPickerProps = AntDMonthPickerProps & React.RefAttributes<DatePickerRef> & DatePickerAccessibilityProps;

function getEmotionStyles(clsPrefix: string, theme: Theme): SerializedStyles {
  const classFocused = `.${clsPrefix}-focused`;
  const classActiveBar = `.${clsPrefix}-active-bar`;
  const classSeparator = `.${clsPrefix}-separator`;
  const classSuffix = `.${clsPrefix}-suffix`;

  const styles: CSSObject = {
    height: 32,
    borderRadius: theme.legacyBorders.borderRadiusMd,
    borderColor: theme.colors.border,
    color: theme.colors.textPrimary,
    transition: 'border 0s, box-shadow 0s',
    [`&${classFocused},:hover`]: {
      borderColor: theme.colors.actionDefaultBorderHover,
    },
    '&:active': {
      borderColor: theme.colors.actionDefaultBorderPress,
    },
    [`&${classFocused}`]: {
      boxShadow: `none !important`,
      outline: `${theme.colors.actionDefaultBorderFocus} solid 2px !important`,
      outlineOffset: '-2px !important',
      borderColor: 'transparent !important',
    },
    [`& ${classActiveBar}`]: {
      background: `${theme.colors.actionDefaultBorderPress} !important`,
    },
    [`& input::placeholder, & ${classSeparator}, & ${classSuffix}`]: {
      color: theme.colors.textPrimary,
    },
  };

  return css(styles);
}

const getDropdownStyles = (theme: Theme) => {
  return {
    zIndex: theme.options.zIndexBase + 50,
  };
};

function useDatePickerStyles(): SerializedStyles {
  const { theme, getPrefixedClassName } = useDesignSystemTheme();
  const clsPrefix = getPrefixedClassName('picker');

  return getEmotionStyles(clsPrefix, theme);
}

const AccessibilityWrapper: React.FC<{
  children: React.ReactNode;
  ariaLive?: 'assertive' | 'polite';
}> = ({ children, ariaLive = 'assertive', ...restProps }) => {
  const { theme } = useDesignSystemTheme();
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (ref.current) {
      const inputs = theme.isDarkMode
        ? ref.current.querySelectorAll('.du-bois-dark-picker-input > input')
        : ref.current.querySelectorAll('.du-bois-light-picker-input > input');

      inputs.forEach((input) => input.setAttribute('aria-live', ariaLive));
    }
  }, [ref, ariaLive, theme.isDarkMode]);

  return (
    <div {...restProps} ref={ref}>
      {children}
    </div>
  );
};

export const DuboisDatePicker: React.VFC<DatePickerProps> = forwardRef(
  (props: DatePickerProps, ref: React.ForwardedRef<DatePickerRef>) => {
    const styles = useDatePickerStyles();
    const { theme } = useDesignSystemTheme();

    const { ariaLive, wrapperDivProps, ...restProps } = props;

    return (
      <DesignSystemAntDConfigProvider>
        <AccessibilityWrapper {...addDebugOutlineIfEnabled()} {...wrapperDivProps} ariaLive={ariaLive}>
          <AntDDatePicker
            css={styles}
            ref={ref as any}
            {...restProps}
            popupStyle={{ ...getDropdownStyles(theme), ...(props.popupStyle || {}) }}
          />
        </AccessibilityWrapper>
      </DesignSystemAntDConfigProvider>
    );
  },
);

const RangePicker: React.VFC<RangePickerProps> = forwardRef(
  (props: RangePickerProps, ref: React.ForwardedRef<DatePickerRef>) => {
    const styles = useDatePickerStyles();
    const { theme } = useDesignSystemTheme();
    const { ariaLive, wrapperDivProps, ...restProps } = props;

    return (
      <DesignSystemAntDConfigProvider>
        <AccessibilityWrapper {...wrapperDivProps} ariaLive={ariaLive}>
          <AntDDatePicker.RangePicker
            {...addDebugOutlineIfEnabled()}
            css={styles}
            {...restProps}
            ref={ref as any}
            popupStyle={{ ...getDropdownStyles(theme), ...(props.popupStyle || {}) }}
          />
        </AccessibilityWrapper>
      </DesignSystemAntDConfigProvider>
    );
  },
);

const TimePicker: React.VFC<TimePickerProps> = forwardRef(
  (props: TimePickerProps, ref: React.ForwardedRef<DatePickerRef>) => {
    const styles = useDatePickerStyles();
    const { theme } = useDesignSystemTheme();
    const { ariaLive, wrapperDivProps, ...restProps } = props;

    return (
      <DesignSystemAntDConfigProvider>
        <AccessibilityWrapper {...addDebugOutlineIfEnabled()} {...wrapperDivProps} ariaLive={ariaLive}>
          <AntDDatePicker.TimePicker
            css={styles}
            {...restProps}
            ref={ref as any}
            popupStyle={{ ...getDropdownStyles(theme), ...(props.popupStyle || {}) }}
          />
        </AccessibilityWrapper>
      </DesignSystemAntDConfigProvider>
    );
  },
);

const QuarterPicker: React.VFC<QuarterPickerProps> = forwardRef(
  (props: QuarterPickerProps, ref: React.ForwardedRef<DatePickerRef>) => {
    const styles = useDatePickerStyles();
    const { theme } = useDesignSystemTheme();
    const { ariaLive, wrapperDivProps, ...restProps } = props;

    return (
      <DesignSystemAntDConfigProvider>
        <AccessibilityWrapper {...addDebugOutlineIfEnabled()} {...wrapperDivProps} ariaLive={ariaLive}>
          <AntDDatePicker.QuarterPicker
            css={styles}
            {...restProps}
            ref={ref as any}
            popupStyle={{ ...getDropdownStyles(theme), ...(props.popupStyle || {}) }}
          />
        </AccessibilityWrapper>
      </DesignSystemAntDConfigProvider>
    );
  },
);

const WeekPicker: React.VFC<WeekPickerProps> = forwardRef(
  (props: WeekPickerProps, ref: React.ForwardedRef<DatePickerRef>) => {
    const styles = useDatePickerStyles();
    const { theme } = useDesignSystemTheme();
    const { ariaLive, wrapperDivProps, ...restProps } = props;

    return (
      <DesignSystemAntDConfigProvider>
        <AccessibilityWrapper {...addDebugOutlineIfEnabled()} {...wrapperDivProps} ariaLive={ariaLive}>
          <AntDDatePicker.WeekPicker
            css={styles}
            {...restProps}
            ref={ref as any}
            popupStyle={{ ...getDropdownStyles(theme), ...(props.popupStyle || {}) }}
          />
        </AccessibilityWrapper>
      </DesignSystemAntDConfigProvider>
    );
  },
);

const MonthPicker: React.VFC<MonthPickerProps> = forwardRef(
  (props: MonthPickerProps, ref: React.ForwardedRef<DatePickerRef>) => {
    const styles = useDatePickerStyles();
    const { theme } = useDesignSystemTheme();

    const { ariaLive, wrapperDivProps, ...restProps } = props;

    return (
      <DesignSystemAntDConfigProvider>
        <AccessibilityWrapper {...addDebugOutlineIfEnabled()} {...wrapperDivProps} ariaLive={ariaLive}>
          <AntDDatePicker.MonthPicker
            css={styles}
            {...restProps}
            ref={ref as any}
            popupStyle={{ ...getDropdownStyles(theme), ...(props.popupStyle || {}) }}
          />
        </AccessibilityWrapper>
      </DesignSystemAntDConfigProvider>
    );
  },
);

const YearPicker: React.VFC<YearPickerProps> = forwardRef(
  (props: YearPickerProps, ref: React.ForwardedRef<DatePickerRef>) => {
    const styles = useDatePickerStyles();
    const { theme } = useDesignSystemTheme();

    const { ariaLive, wrapperDivProps, ...restProps } = props;

    return (
      <DesignSystemAntDConfigProvider>
        <AccessibilityWrapper {...addDebugOutlineIfEnabled()} {...wrapperDivProps} ariaLive={ariaLive}>
          <AntDDatePicker.YearPicker
            css={styles}
            {...restProps}
            ref={ref as any}
            popupStyle={{ ...getDropdownStyles(theme), ...(props.popupStyle || {}) }}
          />
        </AccessibilityWrapper>
      </DesignSystemAntDConfigProvider>
    );
  },
);

/**
 * `LegacyDatePicker` was added as a temporary solution pending an
 * official Du Bois replacement. Use with caution.
 * @deprecated
 */
export const LegacyDatePicker = /* #__PURE__ */ Object.assign(DuboisDatePicker, {
  /**
   * See deprecation notice for `LegacyDatePicker`.
   * @deprecated
   */
  RangePicker,
  /**
   * See deprecation notice for `LegacyDatePicker`.
   * @deprecated
   */
  TimePicker,
  /**
   * See deprecation notice for `LegacyDatePicker`.
   * @deprecated
   */
  QuarterPicker,
  /**
   * See deprecation notice for `LegacyDatePicker`.
   * @deprecated
   */
  WeekPicker,
  /**
   * See deprecation notice for `LegacyDatePicker`.
   * @deprecated
   */
  MonthPicker,
  /**
   * See deprecation notice for `LegacyDatePicker`.
   * @deprecated
   */
  YearPicker,
});
