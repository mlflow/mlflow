import {
  endOfToday,
  endOfYesterday,
  format as formatDateFns,
  isAfter,
  isBefore,
  isValid,
  startOfToday,
  startOfWeek,
  startOfYesterday,
  sub,
} from 'date-fns';
import type { ForwardedRef, HTMLAttributes, KeyboardEvent, MouseEvent } from 'react';
import { forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react';
import { Button as DayPickerButton, DayPicker, useDayRender } from 'react-day-picker';
import type {
  DateAfter,
  DateBefore,
  DateRange as DayPickerDateRange,
  DayPickerRangeProps,
  DayPickerSingleProps,
  DayProps,
  Matcher,
} from 'react-day-picker';

import { getDayPickerStyles } from './styles';
import { generateDatePickerClassNames } from './utils';
import type { InputProps } from '../../design-system';
import {
  Button,
  ChevronLeftIcon,
  ChevronRightIcon,
  ClockIcon,
  Input,
  Popover,
  useDesignSystemTheme,
} from '../../design-system';
import type { HTMLDataAttributes, ValidationState } from '../../design-system/types';

const handleInputKeyDown = (event: React.KeyboardEvent<HTMLInputElement>, setIsVisible: (open: boolean) => void) => {
  if (event.key === ' ' || event.key === 'Enter' || event.key === 'Space') {
    event.preventDefault();
    event.stopPropagation();
    setIsVisible(true);
  }
};

export interface DatePickerChangeEventType {
  target: {
    name?: string;
    value: Date | undefined;
  };
  type: string;
  updateLocation: 'input' | 'calendar';
}

export interface DatePickerWrapperProps {
  wrapperProps?: HTMLAttributes<HTMLDivElement> & HTMLDataAttributes;
}

export interface DatePickerProps
  extends Omit<InputProps, 'type' | 'suffix' | 'onKeyDown' | 'value' | 'onChange'>,
    DatePickerWrapperProps {
  onChange?: (e: DatePickerChangeEventType) => void;
  onClear?: () => void;
  open?: boolean;
  onOpenChange?: (visible: boolean) => void;
  value?: Date;
  validationState?: ValidationState;
  includeTime?: boolean;
  /**
   * Expected format HH:mm
   */
  defaultTime?: string;
  datePickerProps?: Omit<DayPickerSingleProps, 'mode' | 'selected'> | Omit<DayPickerRangeProps, 'mode' | 'selected'>;
  timeInputProps?: Omit<InputProps, 'type' | 'allowClear' | 'onChange' | 'value' | 'componentId'>;
  name?: string;
  width?: string | number;
  maxWidth?: string | number;
  minWidth?: string | number;
  dateTimeDisabledFn?: (date: Date) => boolean;
  quickActions?: DatePickerQuickActionProps[];
  onOkPress?: () => void;
  okButtonLabel?: string;
  /**
   * DO NOT USE THIS PROP. This is only for internal use.
   */
  showTimeZone?: boolean;
  /**
   * Custom timezone label, this has no functional impact, converting to the correct timezone must be done outside this component
   */
  customTimeZoneLabel?: string;
}

function Day(props: DayProps): JSX.Element {
  const buttonRef = useRef<HTMLButtonElement>(null);
  const dayRender = useDayRender(props.date, props.displayMonth, buttonRef);

  if (dayRender.isHidden) {
    return <div role="cell" />;
  }
  if (!dayRender.isButton) {
    return <div {...dayRender.divProps} />;
  }
  return <DayPickerButton name="day" ref={buttonRef} {...dayRender.buttonProps} role="button" />;
}

interface DatePickerQuickActionProps {
  label: string;
  /**
   * Do not pass Date[] as value, it's only for internal use
   */
  value: Date | Date[];
  onClick?: (value: Date | Date[]) => void;
}

export const getDatePickerQuickActionBasic = ({
  today,
  yesterday,
  sevenDaysAgo,
}: {
  today?: Partial<DatePickerQuickActionProps>;
  yesterday?: Partial<DatePickerQuickActionProps>;
  sevenDaysAgo?: Partial<DatePickerQuickActionProps>;
}): DatePickerQuickActionProps[] => [
  {
    label: 'Today',
    value: startOfToday(),
    ...today,
  },
  {
    label: 'Yesterday',
    value: startOfYesterday(),
    ...yesterday,
  },
  {
    label: '7 days ago',
    value: sub(startOfToday(), { days: 7 }),
    ...sevenDaysAgo,
  },
];

export const DatePicker = forwardRef<HTMLInputElement, DatePickerProps>(
  (props: DatePickerProps, ref: ForwardedRef<HTMLInputElement>) => {
    const { classNamePrefix, theme } = useDesignSystemTheme();
    const {
      id,
      name,
      value,
      validationState,
      onChange,
      allowClear,
      onClear,
      includeTime,
      defaultTime,
      onOpenChange,
      open,
      datePickerProps,
      timeInputProps,
      mode = 'single',
      selected,
      width,
      maxWidth,
      minWidth,
      dateTimeDisabledFn,
      quickActions,
      wrapperProps,
      onOkPress,
      okButtonLabel,
      showTimeZone,
      customTimeZoneLabel,
      ...restProps
    } = props as DatePickerProps & { selected?: DayPickerDateRange; mode: 'single' | 'range' };
    const format = includeTime ? 'yyyy-MM-dd HH:mm' : 'yyyy-MM-dd';
    const [date, setDate] = useState<Date | undefined>(value);
    const [timezone, setTimezone] = useState<string | undefined>(customTimeZoneLabel);

    const [isVisible, setIsVisible] = useState(Boolean(open));
    const inputRef = useRef(null);

    const visibleRef = useRef(isVisible);
    // Needed to avoid the clear icon click also reopening the datepicker
    const fromClearRef = useRef(null);

    useEffect(() => {
      if (!isVisible && visibleRef.current) {
        (inputRef.current as any)?.focus();
      }
      visibleRef.current = isVisible;
      onOpenChange?.(isVisible);
    }, [isVisible, onOpenChange]);

    useEffect(() => {
      setIsVisible(Boolean(open));
    }, [open]);

    useEffect(() => {
      const now = new Date();
      if (showTimeZone) {
        if (customTimeZoneLabel) {
          setTimezone(customTimeZoneLabel);
          return;
        }

        setTimezone(
          Intl.DateTimeFormat('en-US', {
            timeZoneName: 'short',
          })
            .formatToParts(now)
            .find((part) => part.type === 'timeZoneName')?.value ?? formatDateFns(now, 'z'),
        );
      } else {
        setTimezone(undefined);
      }
    }, [showTimeZone, customTimeZoneLabel]);

    useEffect(() => {
      if (value) {
        if (value instanceof Date && isValid(value)) {
          setDate(value);
        } else {
          if (isValid(new Date(value))) {
            setDate(new Date(value));
          } else {
            setDate(undefined);
          }
        }
      } else {
        setDate(undefined);
      }
    }, [value]);

    const handleChange = useCallback(
      (date: Date | undefined, isCalendarUpdate: boolean) => {
        if (onChange) {
          onChange({
            target: {
              name,
              value: date,
            },
            type: 'change',
            updateLocation: isCalendarUpdate ? 'calendar' : 'input',
          });
        }
      },
      [onChange, name],
    );

    const handleDatePickerUpdate = (date: Date) => {
      setDate((prevDate) => {
        // Set default time if date is set the first time
        if (!prevDate && date && includeTime && defaultTime) {
          const timeSplit = defaultTime.split(':');
          date?.setHours(+timeSplit[0]);
          date?.setMinutes(+timeSplit[1]);
        } else if (prevDate && date && includeTime) {
          date.setHours(prevDate.getHours());
          date.setMinutes(prevDate.getMinutes());
        }
        handleChange?.(date, true);
        return date;
      });

      if (!includeTime) {
        setIsVisible(false);
      }
    };

    const handleInputUpdate = (updatedDate: Date) => {
      if (!updatedDate || !isValid(updatedDate)) {
        setDate(undefined);
        handleChange?.(undefined, false);
        return;
      }

      if (date && updatedDate && includeTime) {
        updatedDate.setHours(updatedDate.getHours());
        updatedDate.setMinutes(updatedDate.getMinutes());
      }

      setDate(updatedDate);
      handleChange?.(updatedDate, false);

      if (!includeTime) {
        setIsVisible(false);
      }
    };

    const handleClear = useCallback(() => {
      setDate(undefined);
      onClear?.();
      handleChange?.(undefined, false);
    }, [onClear, handleChange]);

    const handleTimeUpdate = (e: any) => {
      const newTime = e.nativeEvent?.target?.value;

      const time = date && isValid(date) ? formatDateFns(date, 'HH:mm') : undefined;
      if (newTime && newTime !== time) {
        if (date) {
          const updatedDate = new Date(date);
          const timeSplit = newTime.split(':');
          updatedDate.setHours(+timeSplit[0]);
          updatedDate.setMinutes(+timeSplit[1]);
          handleInputUpdate(updatedDate);
        }
      }
    };

    // Manually add the clear icon click event listener to avoid reopening the datepicker when clearing the input
    useEffect(() => {
      if (allowClear && inputRef.current) {
        const clearIcon = (inputRef.current as any).input
          ?.closest('[type="button"]')
          ?.querySelector(`.${classNamePrefix}-input-clear-icon`);

        if (clearIcon !== fromClearRef.current) {
          fromClearRef.current = clearIcon;
          const clientEventListener = (e: MouseEvent<HTMLSpanElement>) => {
            e.stopPropagation();
            e.preventDefault();
            handleClear();
          };

          clearIcon.addEventListener('click', clientEventListener);
        }
      }
    }, [classNamePrefix, defaultTime, handleClear, allowClear]);

    const { classNames, datePickerStyles } = useMemo(
      () => ({
        classNames: generateDatePickerClassNames(`${classNamePrefix}-datepicker`),
        datePickerStyles: getDayPickerStyles(`${classNamePrefix}-datepicker`, theme),
      }),
      [classNamePrefix, theme],
    );

    const chevronLeftIconComp = (props: any) => <ChevronLeftIcon {...props} />;
    const chevronRightIconComp = (props: any) => <ChevronRightIcon {...props} />;

    return (
      <div
        className={`${classNamePrefix}-datepicker`}
        css={{ width, minWidth, maxWidth, pointerEvents: restProps?.disabled ? 'none' : 'auto' }}
        {...wrapperProps}
      >
        <Popover.Root
          componentId="codegen_design-system_src_development_datepicker_datepicker.tsx_330"
          open={isVisible}
          onOpenChange={setIsVisible}
        >
          <Popover.Trigger asChild disabled={restProps?.disabled} role="combobox">
            <div>
              <Input
                id={id}
                ref={inputRef}
                name={name}
                validationState={validationState}
                allowClear={allowClear}
                placeholder="Select Date"
                aria-label={includeTime ? 'Select Date and Time' : 'Select Date'}
                prefix="Date:"
                role="textbox"
                suffix={showTimeZone ? <span>{timezone}</span> : undefined}
                max={includeTime ? '9999-12-31T23:59' : '9999-12-31'}
                {...restProps}
                css={{
                  '*::-webkit-calendar-picker-indicator': { display: 'none' },
                  [`.${classNamePrefix}-input-prefix`]: {
                    ...(!restProps?.disabled && { color: `${theme.colors.textPrimary} !important` }),
                  },
                  [`&.${classNamePrefix}-input-affix-wrapper > *`]: {
                    height: theme.typography.lineHeightBase,
                  },
                  ...(showTimeZone && {
                    [`.${classNamePrefix}-input-suffix`]: {
                      display: 'inline-flex',
                      flexDirection: 'row-reverse',
                      gap: theme.spacing.sm,
                      alignItems: 'center',
                    },
                  }),
                }}
                type={includeTime ? 'datetime-local' : 'date'}
                onKeyDown={(event) => handleInputKeyDown(event, setIsVisible)}
                onChange={(e) => handleInputUpdate(new Date(e.target.value))}
                value={date && isValid(date) ? formatDateFns(date, format) : undefined}
              />
              <input type="hidden" ref={ref} value={(date as any) || ''} />
            </div>
          </Popover.Trigger>
          <Popover.Content align="start" css={datePickerStyles}>
            <DayPicker
              initialFocus
              {...datePickerProps}
              mode={mode as any}
              selected={mode === 'range' ? selected : date}
              onDayClick={handleDatePickerUpdate}
              showOutsideDays={mode === 'range' ? false : true}
              formatters={{
                formatWeekdayName: (date) => formatDateFns(date, 'iiiii', { locale: datePickerProps?.locale }),
              }}
              components={{
                Day,
                IconLeft: chevronLeftIconComp,
                IconRight: chevronRightIconComp,
              }}
              defaultMonth={date}
              classNames={classNames}
            />
            {quickActions?.length && (
              <div
                style={{
                  display: 'flex',
                  gap: theme.spacing.sm,
                  marginBottom: theme.spacing.md,
                  padding: `${theme.spacing.xs}px ${theme.spacing.xs}px 0`,
                  maxWidth: 225,
                  flexWrap: 'wrap',
                }}
              >
                {quickActions?.map((action, i) => (
                  <Button
                    size="small"
                    componentId="codegen_design-system_src_design-system_datepicker_datepicker.tsx_281"
                    key={i}
                    onClick={() =>
                      action.onClick
                        ? action.onClick(action.value)
                        : !Array.isArray(action.value) && handleDatePickerUpdate(action.value)
                    }
                  >
                    {action.label}
                  </Button>
                ))}
              </div>
            )}
            {includeTime && (
              <Input
                componentId="codegen_design-system_src_development_datepicker_datepicker.tsx_306"
                type="time"
                aria-label="Time"
                role="textbox"
                {...timeInputProps}
                value={date && isValid(date) ? formatDateFns(date, 'HH:mm') : undefined}
                onChange={handleTimeUpdate}
                css={{
                  '*::-webkit-calendar-picker-indicator': {
                    position: 'absolute',
                    right: -8,
                    width: theme.general.iconSize,
                    height: theme.general.iconSize,
                    zIndex: theme.options.zIndexBase + 1,
                    color: 'transparent',
                    background: 'transparent',
                  },
                  [`.${classNamePrefix}-input-suffix`]: {
                    position: 'absolute',
                    right: 12,
                    top: 8,
                  },
                }}
                suffix={<ClockIcon />}
                disabled={timeInputProps?.disabled}
              />
            )}
            {mode === 'range' && includeTime && onOkPress && (
              <div css={{ paddingTop: theme.spacing.md, display: 'flex', justifyContent: 'flex-end' }}>
                <Button
                  aria-label="Open end date picker"
                  type="primary"
                  componentId="datepicker-dubois-ok-button"
                  onClick={onOkPress}
                >
                  {okButtonLabel ?? 'Ok'}
                </Button>
              </div>
            )}
          </Popover.Content>
        </Popover.Root>
      </div>
    );
  },
);

export interface RangePickerProps extends Omit<DayPickerRangeProps, 'mode'>, DatePickerWrapperProps {
  onChange?: (date: DayPickerDateRange | undefined) => void;
  startDatePickerProps?: DatePickerProps & { ref?: ForwardedRef<HTMLInputElement> };
  endDatePickerProps?: DatePickerProps & { ref?: ForwardedRef<HTMLInputElement> };
  includeTime?: boolean;
  allowClear?: boolean;
  /**
   * Minimum recommended width 300px without `includeTime` and 350px with `includeTime`. 400px if both `includeTime` and `allowClear` are true
   */
  width?: string | number;
  /**
   * Minimum recommended width 300px without `includeTime` and 350px with `includeTime`. 400px if both `includeTime` and `allowClear` are true
   */
  maxWidth?: string | number;
  /**
   * Minimum recommended width 300px without `includeTime` and 350px with `includeTime`. 400px if both `includeTime` and `allowClear` are true
   */
  minWidth?: string | number;
  disabled?: boolean;
  quickActions?: RangePickerQuickActionProps[];
}

export interface DateRange extends DayPickerDateRange {}

interface RangePickerQuickActionProps extends DatePickerQuickActionProps {}

export const getRangeQuickActionsBasic = ({
  today,
  yesterday,
  lastWeek,
}: {
  today?: Partial<RangePickerQuickActionProps>;
  yesterday?: Partial<RangePickerQuickActionProps>;
  lastWeek?: Partial<RangePickerQuickActionProps>;
}): RangePickerQuickActionProps[] => {
  const todayStart = startOfToday();
  const weekStart = startOfWeek(todayStart);
  return [
    {
      label: 'Today',
      value: [todayStart, endOfToday()],
      ...today,
    },
    {
      label: 'Yesterday',
      value: [startOfYesterday(), endOfYesterday()],
      ...yesterday,
    },
    {
      label: 'Last week',
      value: [sub(weekStart, { days: 7 }), sub(weekStart, { days: 1 })],
      ...lastWeek,
    },
  ];
};

export const RangePicker = (props: RangePickerProps) => {
  const {
    id,
    onChange,
    startDatePickerProps,
    endDatePickerProps,
    includeTime,
    allowClear,
    minWidth,
    maxWidth,
    width,
    disabled,
    quickActions,
    wrapperProps,
  } = props;
  const [range, setRange] = useState<DayPickerDateRange | undefined>({
    from: startDatePickerProps?.value,
    to: endDatePickerProps?.value,
  });
  const { classNamePrefix } = useDesignSystemTheme();

  // Focus is lost when the popover is closed, we need to set the focus back to the input that opened the popover manually.
  const [isFromVisible, setIsFromVisible] = useState(false);
  const [isToVisible, setIsToVisible] = useState(false);
  const [isRangeInputFocused, setIsRangeInputFocused] = useState(false);

  const fromInputRef = useRef<HTMLInputElement>(null);
  const toInputRef = useRef<HTMLInputElement>(null);

  useImperativeHandle(startDatePickerProps?.ref, () => fromInputRef.current as any);
  useImperativeHandle(endDatePickerProps?.ref, () => toInputRef.current as any);

  const fromInputRefVisible = useRef(isFromVisible);
  const toInputRefVisible = useRef(isToVisible);

  useEffect(() => {
    if (!isFromVisible && fromInputRefVisible.current) {
      (fromInputRef.current as any)?.focus();
    }
    fromInputRefVisible.current = isFromVisible;
  }, [isFromVisible]);

  useEffect(() => {
    if (!isToVisible && toInputRefVisible.current) {
      (toInputRef.current as any)?.focus();
    }
    toInputRefVisible.current = isToVisible;
  }, [isToVisible]);

  const checkIfDateTimeIsDisabled = useCallback(
    (date, isStart = false): boolean => {
      const dateToCompareTo = isStart ? range?.to : range?.from;
      if (date && dateToCompareTo) {
        return isStart ? isAfter(date, dateToCompareTo) : isBefore(date, dateToCompareTo);
      }

      return false;
    },
    [range],
  );

  useEffect(() => {
    setRange((prevValue) => ({
      from: startDatePickerProps?.value,
      to: prevValue?.to,
    }));
  }, [startDatePickerProps?.value]);

  useEffect(() => {
    setRange((prevValue) => ({
      from: prevValue?.from,
      to: endDatePickerProps?.value,
    }));
  }, [endDatePickerProps?.value]);

  const quickActionsWithHandler = useMemo(() => {
    if (quickActions) {
      return quickActions.map((action) => {
        if (Array.isArray(action.value)) {
          return {
            ...action,
            onClick: ((value: Date[]) => {
              setRange({ from: value[0], to: value[1] });
              onChange?.({ from: value[0], to: value[1] });
              action.onClick?.(value);
              setIsFromVisible(false);
              setIsToVisible(false);
            }) as (value: Date | Date[]) => void,
          };
        }
        return action;
      });
    }

    return quickActions;
  }, [quickActions, onChange]);

  const handleUpdateDate = useCallback(
    (e: DatePickerChangeEventType, isStart: boolean) => {
      const date = e.target.value;
      const newRange: DateRange = isStart ? { from: date, to: range?.to } : { from: range?.from, to: date };
      if (!includeTime) {
        if (isStart) {
          setIsFromVisible(false);
          if (e.updateLocation === 'calendar') {
            setIsToVisible(true);
          }
        } else {
          setIsToVisible(false);
        }
      }

      if (isStart) {
        startDatePickerProps?.onChange?.(e);
      } else {
        endDatePickerProps?.onChange?.(e);
      }

      setRange(newRange);
      onChange?.(newRange);
    },
    [onChange, includeTime, startDatePickerProps, endDatePickerProps, range],
  );

  // Use useMemo to calculate disabled dates
  const disabledDates = useMemo(() => {
    let startDisabledFromProps, endDisabledFromProps;
    if (startDatePickerProps?.datePickerProps?.disabled) {
      startDisabledFromProps = Array.isArray(startDatePickerProps?.datePickerProps?.disabled)
        ? startDatePickerProps?.datePickerProps?.disabled
        : [startDatePickerProps?.datePickerProps?.disabled];
    }
    const startDisabled: Matcher[] = [
      { after: range?.to } as DateAfter,
      ...(startDisabledFromProps ? (startDisabledFromProps as Matcher[]) : []),
    ].filter(Boolean);

    if (endDatePickerProps?.datePickerProps?.disabled) {
      endDisabledFromProps = Array.isArray(endDatePickerProps?.datePickerProps?.disabled)
        ? endDatePickerProps?.datePickerProps?.disabled
        : [endDatePickerProps?.datePickerProps?.disabled];
    }

    const endDisabled: Matcher[] = [
      { before: range?.from } as DateBefore,
      ...(endDisabledFromProps ? (endDisabledFromProps as Matcher[]) : []),
    ].filter(Boolean);

    return { startDisabled, endDisabled };
  }, [
    range?.from,
    range?.to,
    startDatePickerProps?.datePickerProps?.disabled,
    endDatePickerProps?.datePickerProps?.disabled,
  ]);

  const openEndDatePicker = () => {
    setIsFromVisible(false);
    setIsToVisible(true);
  };

  const closeEndDatePicker = () => {
    setIsToVisible(false);
  };

  const handleTimePickerKeyPress = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      openEndDatePicker();
    }
    props.startDatePickerProps?.timeInputProps?.onKeyDown?.(e);
  };

  return (
    <div
      className={`${classNamePrefix}-rangepicker`}
      {...wrapperProps}
      data-focused={isRangeInputFocused}
      css={{ display: 'flex', alignItems: 'center', minWidth, maxWidth, width }}
    >
      <DatePicker
        quickActions={quickActionsWithHandler}
        prefix="Start:"
        open={isFromVisible}
        onOpenChange={setIsFromVisible}
        okButtonLabel="Next"
        {...startDatePickerProps}
        id={id}
        ref={fromInputRef}
        disabled={disabled || startDatePickerProps?.disabled}
        onChange={(e) => handleUpdateDate(e, true)}
        includeTime={includeTime}
        allowClear={allowClear}
        datePickerProps={{
          ...startDatePickerProps?.datePickerProps,
          disabled: disabledDates.startDisabled,
        }}
        timeInputProps={{
          onKeyDown: handleTimePickerKeyPress,
        }}
        // @ts-expect-error - DatePickerProps does not have a mode property in the public API but is needed for this use case
        mode="range"
        selected={range}
        value={range?.from}
        dateTimeDisabledFn={(date: Date) => checkIfDateTimeIsDisabled(date, true)}
        onFocus={(e) => {
          setIsRangeInputFocused(true);
          startDatePickerProps?.onFocus?.(e);
        }}
        onBlur={(e) => {
          setIsRangeInputFocused(false);
          startDatePickerProps?.onBlur?.(e);
        }}
        css={{
          '*::-webkit-calendar-picker-indicator': { display: 'none' },
          borderTopRightRadius: 0,
          borderBottomRightRadius: 0,
        }}
        wrapperProps={{
          style: { width: '50%' },
        }}
        onOkPress={openEndDatePicker}
      />
      <DatePicker
        quickActions={quickActionsWithHandler}
        prefix="End:"
        min={range?.from?.toString()}
        okButtonLabel="Close"
        {...endDatePickerProps}
        ref={toInputRef}
        disabled={disabled || endDatePickerProps?.disabled}
        onChange={(e) => handleUpdateDate(e, false)}
        includeTime={includeTime}
        open={isToVisible}
        onOpenChange={setIsToVisible}
        allowClear={allowClear}
        datePickerProps={{
          ...endDatePickerProps?.datePickerProps,
          disabled: disabledDates.endDisabled,
        }}
        // @ts-expect-error - DatePickerProps does not have a mode property in the public API but is needed for this use case
        mode="range"
        selected={range}
        value={range?.to}
        dateTimeDisabledFn={(date: Date) => checkIfDateTimeIsDisabled(date, false)}
        onFocus={(e) => {
          setIsRangeInputFocused(true);
          startDatePickerProps?.onFocus?.(e);
        }}
        onBlur={(e) => {
          setIsRangeInputFocused(false);
          startDatePickerProps?.onBlur?.(e);
        }}
        css={{
          borderTopLeftRadius: 0,
          borderBottomLeftRadius: 0,
          left: -1,
        }}
        wrapperProps={{
          style: { width: '50%' },
        }}
        onOkPress={closeEndDatePicker}
      />
    </div>
  );
};
