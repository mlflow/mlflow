import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { endOfToday, endOfYesterday, format as formatDateFns, isAfter, isBefore, isValid, startOfToday, startOfWeek, startOfYesterday, sub, } from 'date-fns';
import { forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react';
import { Button as DayPickerButton, DayPicker, useDayRender } from 'react-day-picker';
import { getDayPickerStyles } from './styles';
import { generateDatePickerClassNames } from './utils';
import { Button, ChevronLeftIcon, ChevronRightIcon, ClockIcon, Input, Popover, useDesignSystemTheme, } from '../../design-system';
const handleInputKeyDown = (event, setIsVisible) => {
    if (event.key === ' ' || event.key === 'Enter' || event.key === 'Space') {
        event.preventDefault();
        event.stopPropagation();
        setIsVisible(true);
    }
};
function Day(props) {
    const buttonRef = useRef(null);
    const dayRender = useDayRender(props.date, props.displayMonth, buttonRef);
    if (dayRender.isHidden) {
        return _jsx("div", { role: "cell" });
    }
    if (!dayRender.isButton) {
        return _jsx("div", { ...dayRender.divProps });
    }
    const ariaLabel = props.date.toLocaleDateString(undefined, {
        weekday: 'long',
        year: 'numeric',
        month: 'long',
        day: 'numeric',
    });
    return _jsx(DayPickerButton, { name: "day", ref: buttonRef, ...dayRender.buttonProps, role: "button", "aria-label": ariaLabel });
}
export const getDatePickerQuickActionBasic = ({ today, yesterday, sevenDaysAgo, }) => [
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
export const DatePicker = forwardRef((props, ref) => {
    const { classNamePrefix, theme } = useDesignSystemTheme();
    const { id, name, value, validationState, onChange, allowClear, onClear, includeTime, includeSeconds, defaultTime, onOpenChange, open, datePickerProps, timeInputProps, mode = 'single', selected, width, maxWidth, minWidth, dateTimeDisabledFn, quickActions, wrapperProps, onOkPress, okButtonLabel, showTimeZone, customTimeZoneLabel, ...restProps } = props;
    const format = includeTime ? (includeSeconds ? 'yyyy-MM-dd HH:mm:ss' : 'yyyy-MM-dd HH:mm') : 'yyyy-MM-dd';
    const [date, setDate] = useState(value);
    const [timezone, setTimezone] = useState(customTimeZoneLabel);
    const [isVisible, setIsVisible] = useState(Boolean(open));
    const inputRef = useRef(null);
    const visibleRef = useRef(isVisible);
    // Needed to avoid the clear icon click also reopening the datepicker
    const fromClearRef = useRef(null);
    useEffect(() => {
        if (!isVisible && visibleRef.current) {
            inputRef.current?.focus();
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
            setTimezone(Intl.DateTimeFormat('en-US', {
                timeZoneName: 'short',
            })
                .formatToParts(now)
                .find((part) => part.type === 'timeZoneName')?.value ?? formatDateFns(now, 'z'));
        }
        else {
            setTimezone(undefined);
        }
    }, [showTimeZone, customTimeZoneLabel]);
    useEffect(() => {
        if (value) {
            if (value instanceof Date && isValid(value)) {
                setDate(value);
            }
            else {
                if (isValid(new Date(value))) {
                    setDate(new Date(value));
                }
                else {
                    setDate(undefined);
                }
            }
        }
        else {
            setDate(undefined);
        }
    }, [value]);
    const handleChange = useCallback((date, isCalendarUpdate) => {
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
    }, [onChange, name]);
    const handleDatePickerUpdate = (date) => {
        setDate((prevDate) => {
            // Set default time if date is set the first time
            if (!prevDate && date && includeTime && defaultTime) {
                const timeSplit = defaultTime.split(':');
                date?.setHours(+timeSplit[0]);
                date?.setMinutes(+timeSplit[1]);
                date.setSeconds(timeSplit.length > 2 ? +timeSplit[2] : 0);
            }
            else if (prevDate && date && includeTime) {
                date.setHours(prevDate.getHours());
                date.setMinutes(prevDate.getMinutes());
                if (includeSeconds) {
                    date.setSeconds(prevDate.getSeconds());
                }
            }
            handleChange?.(date, true);
            return date;
        });
        if (!includeTime) {
            setIsVisible(false);
        }
    };
    const handleInputUpdate = (updatedDate) => {
        if (!updatedDate || !isValid(updatedDate)) {
            setDate(undefined);
            handleChange?.(undefined, false);
            return;
        }
        if (date && updatedDate && includeTime) {
            updatedDate.setHours(updatedDate.getHours());
            updatedDate.setMinutes(updatedDate.getMinutes());
            if (includeSeconds) {
                updatedDate.setSeconds(updatedDate.getSeconds());
            }
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
    const handleTimeUpdate = (e) => {
        const newTime = e.nativeEvent?.target?.value;
        const time = date && isValid(date) ? formatDateFns(date, includeSeconds ? 'HH:mm:ss' : 'HH:mm') : undefined;
        if (newTime && newTime !== time) {
            if (date) {
                const updatedDate = new Date(date);
                const timeSplit = newTime.split(':');
                updatedDate.setHours(+timeSplit[0]);
                updatedDate.setMinutes(+timeSplit[1]);
                if (includeSeconds) {
                    updatedDate.setSeconds(+timeSplit[2]);
                }
                handleInputUpdate(updatedDate);
            }
        }
    };
    // Manually add the clear icon click event listener to avoid reopening the datepicker when clearing the input
    useEffect(() => {
        if (allowClear && inputRef.current) {
            const clearIcon = inputRef.current.input
                ?.closest('[type="button"]')
                ?.querySelector(`.${classNamePrefix}-input-clear-icon`);
            if (clearIcon !== fromClearRef.current) {
                fromClearRef.current = clearIcon;
                const clientEventListener = (e) => {
                    e.stopPropagation();
                    e.preventDefault();
                    handleClear();
                };
                clearIcon.addEventListener('click', clientEventListener);
            }
        }
    }, [classNamePrefix, defaultTime, handleClear, allowClear]);
    const { classNames, datePickerStyles } = useMemo(() => ({
        classNames: generateDatePickerClassNames(`${classNamePrefix}-datepicker`),
        datePickerStyles: getDayPickerStyles(`${classNamePrefix}-datepicker`, theme),
    }), [classNamePrefix, theme]);
    const chevronLeftIconComp = (props) => _jsx(ChevronLeftIcon, { ...props });
    const chevronRightIconComp = (props) => _jsx(ChevronRightIcon, { ...props });
    return (_jsx("div", { className: `${classNamePrefix}-datepicker`, css: { width, minWidth, maxWidth, pointerEvents: restProps?.disabled ? 'none' : 'auto' }, ...wrapperProps, children: _jsxs(Popover.Root, { componentId: "codegen_design-system_src_development_datepicker_datepicker.tsx_330", open: isVisible, onOpenChange: setIsVisible, children: [_jsx(Popover.Trigger, { asChild: true, disabled: restProps?.disabled, role: "combobox", children: _jsxs("div", { children: [_jsx(Input, { id: id, ref: inputRef, name: name, validationState: validationState, allowClear: allowClear, placeholder: "Select Date", "aria-label": includeTime ? 'Select Date and Time' : 'Select Date', prefix: "Date:", role: "textbox", suffix: showTimeZone ? _jsx("span", { children: timezone }) : undefined, max: includeTime ? (includeSeconds ? '9999-12-31T23:59:59' : '9999-12-31T23:59') : '9999-12-31', ...restProps, css: {
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
                                }, type: includeTime ? 'datetime-local' : 'date', step: includeTime && includeSeconds ? 1 : undefined, onKeyDown: (event) => handleInputKeyDown(event, setIsVisible), onChange: (e) => handleInputUpdate(new Date(e.target.value)), value: date && isValid(date) ? formatDateFns(date, format) : undefined }), _jsx("input", { type: "hidden", ref: ref, value: date || '' })] }) }), _jsxs(Popover.Content, { align: "start", css: datePickerStyles, children: [_jsx(DayPicker, { initialFocus: true, ...datePickerProps, mode: mode, selected: mode === 'range' ? selected : date, onDayClick: handleDatePickerUpdate, showOutsideDays: mode === 'range' ? false : true, formatters: {
                                formatWeekdayName: (date) => formatDateFns(date, 'iiiii', { locale: datePickerProps?.locale }),
                            }, components: {
                                Day,
                                IconLeft: chevronLeftIconComp,
                                IconRight: chevronRightIconComp,
                            }, defaultMonth: date, classNames: classNames }), quickActions?.length && (_jsx("div", { style: {
                                display: 'flex',
                                gap: theme.spacing.sm,
                                marginBottom: theme.spacing.md,
                                padding: `${theme.spacing.xs}px ${theme.spacing.xs}px 0`,
                                maxWidth: 225,
                                flexWrap: 'wrap',
                            }, children: quickActions?.map((action, i) => (_jsx(Button, { size: "small", componentId: "codegen_design-system_src_design-system_datepicker_datepicker.tsx_281", onClick: () => action.onClick
                                    ? action.onClick(action.value)
                                    : !Array.isArray(action.value) && handleDatePickerUpdate(action.value), children: action.label }, i))) })), includeTime && (_jsx(Input, { componentId: "codegen_design-system_src_development_datepicker_datepicker.tsx_306", type: "time", step: includeSeconds ? 1 : undefined, "aria-label": "Time", role: "textbox", ...timeInputProps, value: date && isValid(date) ? formatDateFns(date, includeSeconds ? 'HH:mm:ss' : 'HH:mm') : undefined, onChange: handleTimeUpdate, css: {
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
                            }, suffix: _jsx(ClockIcon, {}), disabled: timeInputProps?.disabled })), mode === 'range' && includeTime && onOkPress && (_jsx("div", { css: { paddingTop: theme.spacing.md, display: 'flex', justifyContent: 'flex-end' }, children: _jsx(Button, { "aria-label": "Open end date picker", type: "primary", componentId: "datepicker-dubois-ok-button", onClick: onOkPress, children: okButtonLabel ?? 'Ok' }) }))] })] }) }));
});
export const getRangeQuickActionsBasic = ({ today, yesterday, lastWeek, }) => {
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
export const RangePicker = (props) => {
    const { id, onChange, startDatePickerProps, endDatePickerProps, includeTime, includeSeconds, allowClear, minWidth, maxWidth, width, disabled, quickActions, wrapperProps, } = props;
    const [range, setRange] = useState({
        from: startDatePickerProps?.value,
        to: endDatePickerProps?.value,
    });
    const { classNamePrefix } = useDesignSystemTheme();
    // Focus is lost when the popover is closed, we need to set the focus back to the input that opened the popover manually.
    const [isFromVisible, setIsFromVisible] = useState(false);
    const [isToVisible, setIsToVisible] = useState(false);
    const [isRangeInputFocused, setIsRangeInputFocused] = useState(false);
    const fromInputRef = useRef(null);
    const toInputRef = useRef(null);
    useImperativeHandle(startDatePickerProps?.ref, () => fromInputRef.current);
    useImperativeHandle(endDatePickerProps?.ref, () => toInputRef.current);
    const fromInputRefVisible = useRef(isFromVisible);
    const toInputRefVisible = useRef(isToVisible);
    useEffect(() => {
        if (!isFromVisible && fromInputRefVisible.current) {
            fromInputRef.current?.focus();
        }
        fromInputRefVisible.current = isFromVisible;
    }, [isFromVisible]);
    useEffect(() => {
        if (!isToVisible && toInputRefVisible.current) {
            toInputRef.current?.focus();
        }
        toInputRefVisible.current = isToVisible;
    }, [isToVisible]);
    const checkIfDateTimeIsDisabled = useCallback((date, isStart = false) => {
        const dateToCompareTo = isStart ? range?.to : range?.from;
        if (date && dateToCompareTo) {
            return isStart ? isAfter(date, dateToCompareTo) : isBefore(date, dateToCompareTo);
        }
        return false;
    }, [range]);
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
                        onClick: ((value) => {
                            setRange({ from: value[0], to: value[1] });
                            onChange?.({
                                target: { name: props.name, value: { from: value[0], to: value[1] } },
                                type: 'change',
                                updateLocation: 'preset',
                            });
                            action.onClick?.(value);
                            setIsFromVisible(false);
                            setIsToVisible(false);
                        }),
                    };
                }
                return action;
            });
        }
        return quickActions;
    }, [quickActions, onChange, props.name]);
    const handleUpdateDate = useCallback((e, isStart) => {
        const date = e.target.value;
        const newRange = isStart
            ? { from: date, to: range?.to }
            : {
                from: range?.from,
                to: date,
            };
        if (!includeTime) {
            if (isStart) {
                setIsFromVisible(false);
                if (e.updateLocation === 'calendar') {
                    setIsToVisible(true);
                }
            }
            else {
                setIsToVisible(false);
            }
        }
        if (isStart) {
            startDatePickerProps?.onChange?.(e);
        }
        else {
            endDatePickerProps?.onChange?.(e);
        }
        setRange(newRange);
        onChange?.({
            target: { name: props.name, value: newRange },
            type: 'change',
            updateLocation: e.updateLocation,
        });
    }, [onChange, includeTime, startDatePickerProps, endDatePickerProps, range, props.name]);
    // Use useMemo to calculate disabled dates
    const disabledDates = useMemo(() => {
        let startDisabledFromProps, endDisabledFromProps;
        if (startDatePickerProps?.datePickerProps?.disabled) {
            startDisabledFromProps = Array.isArray(startDatePickerProps?.datePickerProps?.disabled)
                ? startDatePickerProps?.datePickerProps?.disabled
                : [startDatePickerProps?.datePickerProps?.disabled];
        }
        const startDisabled = [
            { after: range?.to },
            ...(startDisabledFromProps ? startDisabledFromProps : []),
        ].filter(Boolean);
        if (endDatePickerProps?.datePickerProps?.disabled) {
            endDisabledFromProps = Array.isArray(endDatePickerProps?.datePickerProps?.disabled)
                ? endDatePickerProps?.datePickerProps?.disabled
                : [endDatePickerProps?.datePickerProps?.disabled];
        }
        const endDisabled = [
            { before: range?.from },
            ...(endDisabledFromProps ? endDisabledFromProps : []),
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
    const handleTimePickerKeyPress = (e) => {
        if (e.key === 'Enter') {
            openEndDatePicker();
        }
        props.startDatePickerProps?.timeInputProps?.onKeyDown?.(e);
    };
    return (_jsxs("div", { className: `${classNamePrefix}-rangepicker`, ...wrapperProps, "data-focused": isRangeInputFocused, css: { display: 'flex', alignItems: 'center', minWidth, maxWidth, width }, children: [_jsx(DatePicker, { quickActions: quickActionsWithHandler, prefix: "Start:", open: isFromVisible, onOpenChange: setIsFromVisible, okButtonLabel: "Next", ...startDatePickerProps, id: id, ref: fromInputRef, disabled: disabled || startDatePickerProps?.disabled, onChange: (e) => handleUpdateDate(e, true), includeTime: includeTime, includeSeconds: includeSeconds, allowClear: allowClear, datePickerProps: {
                    ...startDatePickerProps?.datePickerProps,
                    disabled: disabledDates.startDisabled,
                }, timeInputProps: {
                    onKeyDown: handleTimePickerKeyPress,
                }, 
                // @ts-expect-error - DatePickerProps does not have a mode property in the public API but is needed for this use case
                mode: "range", selected: range, value: range?.from, dateTimeDisabledFn: (date) => checkIfDateTimeIsDisabled(date, true), onFocus: (e) => {
                    setIsRangeInputFocused(true);
                    startDatePickerProps?.onFocus?.(e);
                }, onBlur: (e) => {
                    setIsRangeInputFocused(false);
                    startDatePickerProps?.onBlur?.(e);
                }, css: {
                    '*::-webkit-calendar-picker-indicator': { display: 'none' },
                    borderTopRightRadius: 0,
                    borderBottomRightRadius: 0,
                }, wrapperProps: {
                    style: { width: '50%' },
                }, onOkPress: openEndDatePicker }), _jsx(DatePicker, { quickActions: quickActionsWithHandler, prefix: "End:", min: range?.from?.toString(), okButtonLabel: "Close", ...endDatePickerProps, ref: toInputRef, disabled: disabled || endDatePickerProps?.disabled, onChange: (e) => handleUpdateDate(e, false), includeTime: includeTime, includeSeconds: includeSeconds, open: isToVisible, onOpenChange: setIsToVisible, allowClear: allowClear, datePickerProps: {
                    ...endDatePickerProps?.datePickerProps,
                    disabled: disabledDates.endDisabled,
                }, 
                // @ts-expect-error - DatePickerProps does not have a mode property in the public API but is needed for this use case
                mode: "range", selected: range, value: range?.to, dateTimeDisabledFn: (date) => checkIfDateTimeIsDisabled(date, false), onFocus: (e) => {
                    setIsRangeInputFocused(true);
                    startDatePickerProps?.onFocus?.(e);
                }, onBlur: (e) => {
                    setIsRangeInputFocused(false);
                    startDatePickerProps?.onBlur?.(e);
                }, css: {
                    borderTopLeftRadius: 0,
                    borderBottomLeftRadius: 0,
                    left: -1,
                }, wrapperProps: {
                    style: { width: '50%' },
                }, onOkPress: closeEndDatePicker })] }));
};
//# sourceMappingURL=DatePicker.js.map