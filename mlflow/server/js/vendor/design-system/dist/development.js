import { jsx, Fragment, jsxs } from '@emotion/react/jsx-runtime';
import { css, useTheme } from '@emotion/react';
import React__default, { useMemo, forwardRef, useCallback, useState, useRef, useEffect, useImperativeHandle, createContext, useContext, useReducer } from 'react';
import { d as DesignSystemEventProviderAnalyticsEventTypes, u as useDesignSystemTheme, e as useDesignSystemEventComponentCallbacks, f as DesignSystemEventProviderComponentTypes, h as DesignSystemEventProviderComponentSubTypeMap, j as useNotifyOnFirstView, a as addDebugOutlineIfEnabled, T as Typography, p as primitiveColors, W as WarningIcon, k as DangerIcon, B as Button$1, C as CloseIcon, s as safex, R as Root$3, l as Trigger, m as Content, n as ChevronLeftIcon, o as ChevronRightIcon, q as generateUuidV4, r as getComboboxOptionItemWrapperStyles, i as importantify } from './Popover-C9XzCfd9.js';
export { F as Form, t as RhfForm, c as useFormContext } from './Popover-C9XzCfd9.js';
import { M as MegaphoneIcon, u as useCallbackOnceUntilReset, g as getInputStyles, X as XCircleFillIcon, I as Input, C as ClockIcon, S as SearchIcon } from './index-284YKJ8q.js';
import { startOfToday, startOfYesterday, sub, isValid, format, startOfWeek, endOfToday, endOfYesterday, isAfter, isBefore, parseISO } from 'date-fns';
import { DayPicker, useDayRender, Button as Button$2 } from 'react-day-picker';
import classnames from 'classnames';
import { useMergeRefs } from '@floating-ui/react';
import { RadioGroup, RadioGroupItem } from '@radix-ui/react-radio-group';
import * as Progress$1 from '@radix-ui/react-progress';
import * as RadixToolbar from '@radix-ui/react-toolbar';
import 'antd';
import '@ant-design/icons';
import 'lodash/uniqueId';
import '@radix-ui/react-popover';
import '@radix-ui/react-tooltip';
import '@radix-ui/react-tooltip-patch';
import 'lodash/memoize';
import 'lodash/isEqual';
import '@emotion/unitless';
import 'lodash/endsWith';
import 'lodash/isBoolean';
import 'lodash/isNil';
import 'lodash/isNumber';
import 'lodash/isString';
import 'lodash/mapValues';

const { Text, Paragraph } = Typography;
const BANNER_MIN_HEIGHT = 68;
// Max height will allow 2 lines of description (3 lines total)
const BANNER_MAX_HEIGHT = 82;
const useStyles = (props, theme)=>{
    const bannerLevelToBannerColors = {
        info_light_purple: {
            backgroundDefaultColor: theme.isDarkMode ? '#6E2EC729' : '#ECE1FC',
            actionButtonBackgroundHoverColor: theme.colors.actionDefaultBackgroundHover,
            actionButtonBackgroundPressColor: theme.colors.actionDefaultBackgroundPress,
            textColor: theme.colors.actionDefaultTextDefault,
            textHoverColor: '#92A4B38F',
            textPressColor: theme.colors.actionDefaultTextDefault,
            borderDefaultColor: theme.isDarkMode ? '#955CE5' : '#E2D0FB',
            actionBorderColor: '#92A4B38F',
            closeIconColor: theme.isDarkMode ? '#92A4B3' : '#5F7281',
            iconColor: theme.colors.purple,
            actionButtonBorderHoverColor: theme.colors.actionDefaultBorderHover,
            actionButtonBorderPressColor: theme.colors.actionDefaultBorderPress,
            closeIconBackgroundHoverColor: theme.colors.actionTertiaryBackgroundHover,
            closeIconTextHoverColor: theme.colors.actionTertiaryTextHover,
            closeIconBackgroundPressColor: theme.colors.actionDefaultBackgroundPress,
            closeIconTextPressColor: theme.colors.actionTertiaryTextPress
        },
        info_dark_purple: {
            backgroundDefaultColor: theme.isDarkMode ? '#BC92F7DB' : theme.colors.purple,
            actionButtonBackgroundHoverColor: theme.isDarkMode ? '#BC92F7DB' : theme.colors.purple,
            actionButtonBackgroundPressColor: theme.isDarkMode ? '#BC92F7DB' : theme.colors.purple,
            textColor: theme.colors.actionPrimaryTextDefault,
            textHoverColor: theme.colors.actionPrimaryTextHover,
            textPressColor: theme.colors.actionPrimaryTextPress,
            borderDefaultColor: theme.isDarkMode ? '#BC92F7DB' : theme.colors.purple
        },
        // Clean up the experimental info banners
        info: {
            backgroundDefaultColor: theme.isDarkMode ? '#BC92F7DB' : theme.colors.purple,
            actionButtonBackgroundHoverColor: theme.isDarkMode ? '#BC92F7DB' : theme.colors.purple,
            actionButtonBackgroundPressColor: theme.isDarkMode ? '#BC92F7DB' : theme.colors.purple,
            textColor: theme.colors.actionPrimaryTextDefault,
            textHoverColor: theme.colors.actionPrimaryTextHover,
            textPressColor: theme.colors.actionPrimaryTextPress,
            borderDefaultColor: theme.isDarkMode ? '#BC92F7DB' : theme.colors.purple
        },
        // TODO (PLAT-80558, zack.brody) Update hover and press states once we have colors for these
        warning: {
            backgroundDefaultColor: theme.colors.tagLemon,
            actionButtonBackgroundHoverColor: theme.colors.tagLemon,
            actionButtonBackgroundPressColor: theme.colors.tagLemon,
            textColor: primitiveColors.grey800,
            textHoverColor: primitiveColors.grey800,
            textPressColor: primitiveColors.grey800,
            borderDefaultColor: theme.colors.tagLemon
        },
        error: {
            backgroundDefaultColor: theme.colors.actionDangerPrimaryBackgroundDefault,
            actionButtonBackgroundHoverColor: theme.colors.actionDangerPrimaryBackgroundHover,
            actionButtonBackgroundPressColor: theme.colors.actionDangerPrimaryBackgroundPress,
            textColor: theme.colors.actionPrimaryTextDefault,
            textHoverColor: theme.colors.actionPrimaryTextHover,
            textPressColor: theme.colors.actionPrimaryTextPress,
            borderDefaultColor: theme.colors.actionDangerPrimaryBackgroundDefault
        }
    };
    const colorScheme = bannerLevelToBannerColors[props.level];
    return {
        banner: /*#__PURE__*/ css("max-height:", BANNER_MAX_HEIGHT, "px;display:flex;align-items:center;width:100%;padding:8px;box-sizing:border-box;background-color:", colorScheme.backgroundDefaultColor, ";border:1px solid ", colorScheme.borderDefaultColor, ";"),
        iconContainer: /*#__PURE__*/ css("display:flex;color:", colorScheme.iconColor ? colorScheme.iconColor : colorScheme.textColor, ";align-self:", props.description ? 'flex-start' : 'center', ";box-sizing:border-box;max-width:60px;padding-top:4px;padding-bottom:4px;padding-right:", theme.spacing.xs, "px;"),
        mainContent: /*#__PURE__*/ css("flex-direction:column;align-self:", props.description ? 'flex-start' : 'center', ";display:flex;box-sizing:border-box;padding-right:", theme.spacing.sm, "px;padding-top:2px;padding-bottom:2px;min-width:", theme.spacing.lg, "px;width:100%;"),
        messageTextBlock: /*#__PURE__*/ css("display:-webkit-box;-webkit-line-clamp:1;-webkit-box-orient:vertical;overflow:hidden;&&{color:", colorScheme.textColor, ";}"),
        descriptionBlock: /*#__PURE__*/ css("display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;&&{color:", colorScheme.textColor, ";}"),
        rightContainer: /*#__PURE__*/ css("margin-left:auto;display:flex;align-items:center;"),
        closeIconContainer: /*#__PURE__*/ css("display:flex;margin-left:", theme.spacing.xs, "px;box-sizing:border-box;line-height:0;"),
        closeButton: /*#__PURE__*/ css("cursor:pointer;background:none;border:none;margin:0;&&{height:24px !important;width:24px !important;padding:", theme.spacing.xs, "px !important;box-shadow:unset !important;}&&:hover{background-color:transparent !important;border-color:", colorScheme.textHoverColor, "!important;color:", colorScheme.closeIconTextHoverColor ? colorScheme.closeIconTextHoverColor : colorScheme.textColor, "!important;background-color:", colorScheme.closeIconBackgroundHoverColor ? colorScheme.closeIconBackgroundHoverColor : colorScheme.backgroundDefaultColor, "!important;}&&:active{border-color:", colorScheme.actionBorderColor, "!important;color:", colorScheme.closeIconTextPressColor ? colorScheme.closeIconTextPressColor : colorScheme.textColor, "!important;background-color:", colorScheme.closeIconBackgroundPressColor ? colorScheme.closeIconBackgroundPressColor : colorScheme.backgroundDefaultColor, "!important;}"),
        closeIcon: /*#__PURE__*/ css("color:", colorScheme.closeIconColor ? colorScheme.closeIconColor : colorScheme.textColor, "!important;"),
        actionButtonContainer: /*#__PURE__*/ css("margin-right:", theme.spacing.xs, "px;"),
        // Override design system colors to show the use the action text color for text and border.
        // Also overrides text for links.
        actionButton: /*#__PURE__*/ css("color:", colorScheme.textColor, "!important;border-color:", colorScheme.actionBorderColor ? colorScheme.actionBorderColor : colorScheme.textColor, "!important;box-shadow:unset !important;&:focus,&:hover{border-color:", colorScheme.actionButtonBorderHoverColor ? colorScheme.actionButtonBorderHoverColor : colorScheme.textHoverColor, "!important;color:", colorScheme.textColor, "!important;background-color:", colorScheme.actionButtonBackgroundHoverColor, "!important;}&:active{border-color:", colorScheme.actionButtonBorderPressColor ? colorScheme.actionButtonBorderPressColor : colorScheme.actionBorderColor, "!important;color:", colorScheme.textPressColor, "!important;background-color:", colorScheme.actionButtonBackgroundPressColor, "!important;}a{color:", theme.colors.actionPrimaryTextDefault, ";}a:focus,a:hover{color:", colorScheme.textHoverColor, ";text-decoration:none;}a:active{color:", colorScheme.textPressColor, "        text-decoration:none;}")
    };
};
const levelToIconMap = {
    info: /*#__PURE__*/ jsx(MegaphoneIcon, {
        "data-testid": "level-info-icon"
    }),
    info_light_purple: /*#__PURE__*/ jsx(MegaphoneIcon, {
        "data-testid": "level-info-light-purple-icon"
    }),
    info_dark_purple: /*#__PURE__*/ jsx(MegaphoneIcon, {
        "data-testid": "level-info-dark-purple-icon"
    }),
    warning: /*#__PURE__*/ jsx(WarningIcon, {
        "data-testid": "level-warning-icon"
    }),
    error: /*#__PURE__*/ jsx(DangerIcon, {
        "data-testid": "level-error-icon"
    })
};
const Banner = (props)=>{
    const { level, message, description, ctaText, onAccept, closable, onClose, closeButtonAriaLabel, componentId, analyticsEvents = [
        DesignSystemEventProviderAnalyticsEventTypes.OnView
    ] } = props;
    const [closed, setClosed] = React__default.useState(false);
    const { theme } = useDesignSystemTheme();
    const styles = useStyles(props, theme);
    const memoizedAnalyticsEvents = useMemo(()=>analyticsEvents, [
        analyticsEvents
    ]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.Banner,
        componentId,
        componentSubType: DesignSystemEventProviderComponentSubTypeMap[level],
        analyticsEvents: memoizedAnalyticsEvents
    });
    const { elementRef } = useNotifyOnFirstView({
        onView: eventContext.onView
    });
    const actionButton = onAccept && ctaText ? /*#__PURE__*/ jsx("div", {
        css: styles.actionButtonContainer,
        children: /*#__PURE__*/ jsx(Button$1, {
            componentId: `${componentId}.accept`,
            onClick: onAccept,
            css: styles.actionButton,
            size: "small",
            children: ctaText
        })
    }) : null;
    const close = closable !== false ? /*#__PURE__*/ jsx("div", {
        css: styles.closeIconContainer,
        children: /*#__PURE__*/ jsx(Button$1, {
            componentId: `${componentId}.close`,
            css: styles.closeButton,
            onClick: ()=>{
                if (onClose) {
                    onClose();
                }
                setClosed(true);
            },
            "aria-label": closeButtonAriaLabel ?? 'Close',
            "data-testid": "banner-dismiss",
            children: /*#__PURE__*/ jsx(CloseIcon, {
                css: styles.closeIcon
            })
        })
    }) : null;
    return /*#__PURE__*/ jsx(Fragment, {
        children: !closed && /*#__PURE__*/ jsxs("div", {
            ref: elementRef,
            ...addDebugOutlineIfEnabled(),
            css: styles.banner,
            className: "banner",
            "data-testid": props['data-testid'],
            role: "alert",
            children: [
                /*#__PURE__*/ jsx("div", {
                    css: styles.iconContainer,
                    children: levelToIconMap[level]
                }),
                /*#__PURE__*/ jsxs("div", {
                    css: styles.mainContent,
                    children: [
                        /*#__PURE__*/ jsx(Text, {
                            size: "md",
                            bold: true,
                            css: styles.messageTextBlock,
                            title: message,
                            children: message
                        }),
                        description && /*#__PURE__*/ jsx(Paragraph, {
                            withoutMargins: true,
                            css: styles.descriptionBlock,
                            title: description,
                            children: description
                        })
                    ]
                }),
                /*#__PURE__*/ jsxs("div", {
                    css: styles.rightContainer,
                    children: [
                        actionButton,
                        close
                    ]
                })
            ]
        })
    });
};

const DatePickerInput = /*#__PURE__*/ forwardRef((props, ref)=>{
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.input', false);
    const { theme, classNamePrefix } = useDesignSystemTheme();
    const useNewFormUISpacing = safex('databricks.fe.designsystem.useNewFormUISpacing', false);
    const { componentId, analyticsEvents, showTimeZone, validationState, prefix, suffix, allowClear, onChange, onFocus, onKeyDown, onClear, ...restProps } = props;
    const memoizedAnalyticsEvents = useMemo(()=>analyticsEvents ?? (emitOnView ? [
            DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
            DesignSystemEventProviderAnalyticsEventTypes.OnView
        ] : [
            DesignSystemEventProviderAnalyticsEventTypes.OnValueChange
        ]), [
        analyticsEvents,
        emitOnView
    ]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.Input,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        valueHasNoPii: false
    });
    const { callbackOnceUntilReset: sendAnalyticsEventOncePerFocus, reset: resetSendAnalyticsEventOnFocus } = useCallbackOnceUntilReset(eventContext.onValueChange);
    const handleChange = useCallback((e)=>{
        sendAnalyticsEventOncePerFocus();
        if (e.target.value) {
            onChange?.(e);
        }
    }, [
        onChange,
        sendAnalyticsEventOncePerFocus
    ]);
    const handleClear = useCallback(()=>{
        sendAnalyticsEventOncePerFocus();
        onClear?.();
    }, [
        onClear,
        sendAnalyticsEventOncePerFocus
    ]);
    const handleFocus = useCallback((e)=>{
        resetSendAnalyticsEventOnFocus();
        onFocus?.(e);
    }, [
        onFocus,
        resetSendAnalyticsEventOnFocus
    ]);
    return /*#__PURE__*/ jsxs("div", {
        className: classnames(`${classNamePrefix}-input-affix-wrapper`, restProps.className, restProps.disabled ? `${classNamePrefix}-input-affix-wrapper-disabled` : ''),
        css: [
            getInputStyles(classNamePrefix, theme, {
                validationState,
                type: 'date',
                hasValue: Boolean(restProps.value),
                useNewFormUISpacing
            }, {
                useFocusWithin: true
            }),
            {
                // Chrome, safari & webkit browsers specific fix to hide the calendar picker indicator
                '*::-webkit-calendar-picker-indicator': {
                    display: 'none'
                },
                [`.${classNamePrefix}-input`]: {
                    // vertical alignment fix for all browsers except chrome
                    display: 'inline-flex',
                    // Firefox specific fix to hide the calendar picker indicator
                    paddingRight: '32px !important',
                    marginRight: '-32px',
                    width: 'calc(100% + 32px)',
                    clipPath: 'inset(0 32px 0 0)'
                },
                [`.${classNamePrefix}-input-prefix`]: {
                    ...!restProps?.disabled && {
                        color: `${theme.colors.textPrimary} !important`
                    }
                },
                [`&.${classNamePrefix}-input-affix-wrapper > *`]: {
                    height: theme.typography.lineHeightBase
                },
                ...showTimeZone && {
                    [`.${classNamePrefix}-input-suffix`]: {
                        display: 'inline-flex',
                        flexDirection: 'row-reverse',
                        gap: theme.spacing.sm,
                        alignItems: 'center'
                    }
                }
            }
        ],
        children: [
            prefix && /*#__PURE__*/ jsx("span", {
                className: `${classNamePrefix}-input-prefix`,
                children: prefix
            }),
            /*#__PURE__*/ jsx("input", {
                placeholder: "Select Date",
                ...restProps,
                onChange: handleChange,
                onFocus: handleFocus,
                value: restProps.value ?? '',
                ref: ref,
                className: `${classNamePrefix}-input`,
                onKeyDown: (e)=>{
                    if (e.key === 'Backspace') {
                        handleClear();
                    }
                    onKeyDown?.(e);
                }
            }),
            suffix && /*#__PURE__*/ jsx("span", {
                className: `${classNamePrefix}-input-suffix`,
                children: suffix
            }),
            allowClear && restProps.value && /*#__PURE__*/ jsx("button", {
                "aria-label": "Clear date",
                type: "button",
                css: {
                    background: 'none',
                    border: 'none',
                    display: 'inline-flex',
                    justifyContent: 'center',
                    alignItems: 'center',
                    marginLeft: `${theme.spacing.sm}px !important`,
                    padding: 0
                },
                onClick: (e)=>{
                    e.stopPropagation();
                    e.preventDefault();
                    handleClear();
                },
                children: /*#__PURE__*/ jsx(XCircleFillIcon, {
                    className: `${classNamePrefix}-input-clear-icon`
                })
            })
        ]
    });
});

const getDayPickerStyles = (prefix, theme)=>/*#__PURE__*/ css(".", prefix, "{--rdp-cell-size:", theme.general.heightSm, "px;--rdp-caption-font-size:", theme.typography.fontSizeBase, "px;--rdp-accent-color:", theme.colors.actionPrimaryBackgroundDefault, ";--rdp-background-color:", theme.colors.actionTertiaryBackgroundPress, ";--rdp-accent-color-dark:", theme.colors.actionPrimaryBackgroundDefault, ";--rdp-background-color-dark:", theme.colors.actionTertiaryBackgroundPress, ";--rdp-outline:2px solid var(--rdp-accent-color);--rdp-outline-selected:3px solid var(--rdp-accent-color);--rdp-selected-color:#fff;padding:4px;}.", prefix, "-vhidden{box-sizing:border-box;padding:0;margin:0;background:transparent;border:0;-moz-appearance:none;-webkit-appearance:none;appearance:none;position:absolute !important;top:0;width:1px !important;height:1px !important;padding:0 !important;overflow:hidden !important;clip:rect(1px,1px,1px,1px) !important;border:0 !important;}.", prefix, "-button_reset{appearance:none;position:relative;margin:0;padding:0;cursor:default;color:inherit;background:none;font:inherit;-moz-appearance:none;-webkit-appearance:none;}.", prefix, "-button_reset:focus-visible{outline:none;}.", prefix, "-button{border:2px solid transparent;}.", prefix, "-button[disabled]:not(.", prefix, "-day_selected){opacity:0.25;}.", prefix, "-button:not([disabled]){cursor:pointer;}.", prefix, "-button:focus-visible:not([disabled]){color:inherit;background-color:var(--rdp-background-color);border:var(--rdp-outline);}.", prefix, "-button:hover:not([disabled]):not(.", prefix, "-day_selected){background-color:var(--rdp-background-color);}.", prefix, "-months{display:flex;justify-content:center;}.", prefix, "-month{margin:0 1em;}.", prefix, "-month:first-of-type{margin-left:0;}.", prefix, "-month:last-child{margin-right:0;}.", prefix, "-table{margin:0;max-width:calc(var(--rdp-cell-size) * 7);border-collapse:collapse;}.", prefix, "-with_weeknumber .", prefix, "-table{max-width:calc(var(--rdp-cell-size) * 8);border-collapse:collapse;}.", prefix, "-caption{display:flex;align-items:center;justify-content:space-between;padding:0;text-align:left;}.", prefix, "-multiple_months .", prefix, "-caption{position:relative;display:block;text-align:center;}.", prefix, "-caption_dropdowns{position:relative;display:inline-flex;}.", prefix, "-caption_label{position:relative;z-index:1;display:inline-flex;align-items:center;margin:0;padding:0 0.25em;white-space:nowrap;color:currentColor;border:0;border:2px solid transparent;font-family:inherit;font-size:var(--rdp-caption-font-size);font-weight:600;}.", prefix, "-nav{white-space:nowrap;}.", prefix, "-multiple_months .", prefix, "-caption_start .", prefix, "-nav{position:absolute;top:50%;left:0;transform:translateY(-50%);}.", prefix, "-multiple_months .", prefix, "-caption_end .", prefix, "-nav{position:absolute;top:50%;right:0;transform:translateY(-50%);}.", prefix, "-nav_button{display:inline-flex;align-items:center;justify-content:center;width:var(--rdp-cell-size);height:var(--rdp-cell-size);}.", prefix, "-dropdown_year,.", prefix, "-dropdown_month{position:relative;display:inline-flex;align-items:center;}.", prefix, "-dropdown{appearance:none;position:absolute;z-index:2;top:0;bottom:0;left:0;width:100%;margin:0;padding:0;cursor:inherit;opacity:0;border:none;background-color:transparent;font-family:inherit;font-size:inherit;line-height:inherit;}.", prefix, "-dropdown[disabled]{opacity:unset;color:unset;}.", prefix, "-dropdown:focus-visible:not([disabled]) + .", prefix, "-caption_label{background-color:var(--rdp-background-color);border:var(--rdp-outline);border-radius:6px;}.", prefix, "-dropdown_icon{margin:0 0 0 5px;}.", prefix, "-head{border:0;}.", prefix, "-head_row,.", prefix, "-row{height:100%;}.", prefix, "-head_cell{vertical-align:middle;font-size:inherit;font-weight:400;color:", theme.colors.textSecondary, ";text-align:center;height:100%;height:var(--rdp-cell-size);padding:0;text-transform:uppercase;}.", prefix, "-tbody{border:0;}.", prefix, "-tfoot{margin:0.5em;}.", prefix, "-cell{width:var(--rdp-cell-size);height:100%;height:var(--rdp-cell-size);padding:0;text-align:center;}.", prefix, "-weeknumber{font-size:0.75em;}.", prefix, "-weeknumber,.", prefix, "-day{display:flex;overflow:hidden;align-items:center;justify-content:center;box-sizing:border-box;width:var(--rdp-cell-size);max-width:var(--rdp-cell-size);height:var(--rdp-cell-size);margin:0;border:2px solid transparent;border-radius:", theme.general.borderRadiusBase, "px;}.", prefix, "-day_today:not(.", prefix, "-day_outside){font-weight:bold;}.", prefix, "-day_selected,.", prefix, "-day_selected:focus-visible,.", prefix, "-day_selected:hover{color:var(--rdp-selected-color);opacity:1;background-color:var(--rdp-accent-color);}.", prefix, "-day_outside{pointer-events:none;color:", theme.colors.textSecondary, ";}.", prefix, "-day_selected:focus-visible{outline:var(--rdp-outline);outline-offset:2px;z-index:1;}.", prefix, ":not([dir='rtl']) .", prefix, "-day_range_start:not(.", prefix, "-day_range_end){border-top-right-radius:0;border-bottom-right-radius:0;}.", prefix, ":not([dir='rtl']) .", prefix, "-day_range_end:not(.", prefix, "-day_range_start){border-top-left-radius:0;border-bottom-left-radius:0;}.", prefix, "[dir='rtl'] .", prefix, "-day_range_start:not(.", prefix, "-day_range_end){border-top-left-radius:0;border-bottom-left-radius:0;}.", prefix, "[dir='rtl'] .", prefix, "-day_range_end:not(.", prefix, "-day_range_start){border-top-right-radius:0;border-bottom-right-radius:0;}.", prefix, "-day_range_start,.", prefix, "-day_range_end{border:0;& > span{width:100%;height:100%;display:flex;align-items:center;justify-content:center;border-radius:", theme.general.borderRadiusBase, "px;background-color:var(--rdp-accent-color);color:", theme.colors.white, ";}}.", prefix, "-day_range_end.", prefix, "-day_range_start{border-radius:", theme.general.borderRadiusBase, "px;}.", prefix, "-day_range_middle{border-radius:0;background-color:var(--rdp-background-color);color:", theme.colors.actionDefaultTextDefault, ";&:hover{color:", theme.colors.actionTertiaryTextHover, ";}}.", prefix, "-row > td:last-of-type .", prefix, "-day_range_middle{border-top-right-radius:", theme.general.borderRadiusBase, "px;border-bottom-right-radius:", theme.general.borderRadiusBase, "px;}.", prefix, "-row > td:first-of-type .", prefix, "-day_range_middle{border-top-left-radius:", theme.general.borderRadiusBase, "px;border-bottom-left-radius:", theme.general.borderRadiusBase, "px;}");

const generateDatePickerClassNames = (prefix)=>({
        root: `${prefix}`,
        multiple_months: `${prefix}-multiple_months`,
        with_weeknumber: `${prefix}-with_weeknumber`,
        vhidden: `${prefix}-vhidden`,
        button_reset: `${prefix}-button_reset`,
        button: `${prefix}-button`,
        caption: `${prefix}-caption`,
        caption_start: `${prefix}-caption_start`,
        caption_end: `${prefix}-caption_end`,
        caption_between: `${prefix}-caption_between`,
        caption_label: `${prefix}-caption_label`,
        caption_dropdowns: `${prefix}-caption_dropdowns`,
        dropdown: `${prefix}-dropdown`,
        dropdown_month: `${prefix}-dropdown_month`,
        dropdown_year: `${prefix}-dropdown_year`,
        dropdown_icon: `${prefix}-dropdown_icon`,
        months: `${prefix}-months`,
        month: `${prefix}-month`,
        table: `${prefix}-table`,
        tbody: `${prefix}-tbody`,
        tfoot: `${prefix}-tfoot`,
        head: `${prefix}-head`,
        head_row: `${prefix}-head_row`,
        head_cell: `${prefix}-head_cell`,
        nav: `${prefix}-nav`,
        nav_button: `${prefix}-nav_button`,
        nav_button_previous: `${prefix}-nav_button_previous`,
        nav_button_next: `${prefix}-nav_button_next`,
        nav_icon: `${prefix}-nav_icon`,
        row: `${prefix}-row`,
        weeknumber: `${prefix}-weeknumber`,
        cell: `${prefix}-cell`,
        day: `${prefix}-day`,
        day_today: `${prefix}-day_today`,
        day_outside: `${prefix}-day_outside`,
        day_selected: `${prefix}-day_selected`,
        day_disabled: `${prefix}-day_disabled`,
        day_hidden: `${prefix}-day_hidden`,
        day_range_start: `${prefix}-day_range_start`,
        day_range_end: `${prefix}-day_range_end`,
        day_range_middle: `${prefix}-day_range_middle`
    });

const DEFAULT_MIN_DATE = '1900-01-01 00:00:00';
const DEFAULT_MAX_DATE = '2100-12-31 23:59:59';
// Helps with adding the timezone offset to the date
// This is needed because the datepicker is in the local timezone and
// dates with negative timezone offsets that don't select time will be negatively offset
// leading to the previous day being selected
const correctDateTimezoneOffset = (date)=>{
    const dateWithCorrectedTimezone = new Date(date);
    if (dateWithCorrectedTimezone.getTimezoneOffset() > 0) {
        dateWithCorrectedTimezone.setMinutes(dateWithCorrectedTimezone.getMinutes() + dateWithCorrectedTimezone.getTimezoneOffset());
    }
    return dateWithCorrectedTimezone;
};
const formatDateLimitForInput = (date, includeTime, includeSeconds, defaultDate)=>{
    let limitDate = defaultDate ? new Date(defaultDate) : undefined;
    if (!limitDate) {
        return [
            undefined,
            undefined
        ];
    }
    if (date) {
        if (date instanceof Date && isValid(date)) {
            limitDate = date;
        } else if (isValid(new Date(date))) {
            limitDate = new Date(date);
        }
        if (!includeTime) {
            limitDate = correctDateTimezoneOffset(limitDate);
        }
    }
    if (includeTime) {
        if (includeSeconds) {
            return [
                limitDate,
                format(limitDate, 'yyyy-MM-dd HH:mm:ss')
            ];
        }
        return [
            limitDate,
            format(limitDate, 'yyyy-MM-dd HH:mm')
        ];
    }
    return [
        limitDate,
        format(limitDate, 'yyyy-MM-dd')
    ];
};
const handleInputKeyDown = (event, setIsVisible, backspaceFn)=>{
    if (event.key === ' ' || event.key === 'Enter' || event.key === 'Space') {
        event.preventDefault();
        event.stopPropagation();
        setIsVisible(true);
    } else if (event.key === 'Backspace' && backspaceFn) {
        event.preventDefault();
        event.stopPropagation();
        backspaceFn?.();
    }
};
function Day(props) {
    const buttonRef = useRef(null);
    const dayRender = useDayRender(props.date, props.displayMonth, buttonRef);
    if (dayRender.isHidden) {
        return /*#__PURE__*/ jsx("div", {
            role: "cell"
        });
    }
    if (!dayRender.isButton) {
        return /*#__PURE__*/ jsx("div", {
            ...dayRender.divProps
        });
    }
    const ariaLabel = props.date.toLocaleDateString(undefined, {
        weekday: 'long',
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    });
    return /*#__PURE__*/ jsx(Button$2, {
        name: "day",
        ref: buttonRef,
        ...dayRender.buttonProps,
        role: "button",
        "aria-label": ariaLabel
    });
}
const getDatePickerQuickActionBasic = ({ today, yesterday, sevenDaysAgo })=>[
        {
            label: 'Today',
            value: startOfToday(),
            ...today
        },
        {
            label: 'Yesterday',
            value: startOfYesterday(),
            ...yesterday
        },
        {
            label: '7 days ago',
            value: sub(startOfToday(), {
                days: 7
            }),
            ...sevenDaysAgo
        }
    ];
const DatePicker = /*#__PURE__*/ forwardRef((props, ref)=>{
    const { classNamePrefix, theme } = useDesignSystemTheme();
    const useNewDatePickerInput = safex('databricks.fe.designsystem.useNewDatePickerInput', false);
    const { id, name, value, validationState, onChange, allowClear, onClear, includeTime, includeSeconds, defaultTime, onOpenChange, open, datePickerProps, timeInputProps, mode = 'single', selected, width, maxWidth, minWidth, dateTimeDisabledFn, quickActions, wrapperProps, onOkPress, okButtonLabel, showTimeZone, customTimeZoneLabel, min, max, ...restProps } = props;
    const format$1 = includeTime ? includeSeconds ? 'yyyy-MM-dd HH:mm:ss' : 'yyyy-MM-dd HH:mm' : 'yyyy-MM-dd';
    const [date, setDate] = useState(value);
    const [timezone, setTimezone] = useState(customTimeZoneLabel);
    const [isVisible, setIsVisible] = useState(Boolean(open));
    // Used to prevent the blur update when the datepicker is opened/closed
    const [preventBlurUpdate, setPreventBlurUpdate] = useState(false);
    const inputRef = useRef(null);
    const visibleRef = useRef(isVisible);
    // Needed to avoid the clear icon click also reopening the datepicker
    const fromClearRef = useRef(null);
    const setDateInputValue = useCallback((date)=>{
        const formattedDate = date && isValid(date) ? format(date, format$1) : undefined;
        if (inputRef.current) {
            inputRef.current.value = formattedDate;
        }
    }, [
        format$1
    ]);
    const setDateAndDateInputValue = useCallback((date)=>{
        setDate(date);
        if (useNewDatePickerInput) {
            setDateInputValue(date);
        }
    }, [
        useNewDatePickerInput,
        setDateInputValue
    ]);
    useEffect(()=>{
        if (!isVisible && visibleRef.current) {
            inputRef.current?.focus();
        }
        visibleRef.current = isVisible;
        onOpenChange?.(isVisible);
    }, [
        isVisible,
        onOpenChange
    ]);
    useEffect(()=>{
        setIsVisible(Boolean(open));
    }, [
        open
    ]);
    useEffect(()=>{
        const now = new Date();
        if (showTimeZone) {
            if (customTimeZoneLabel) {
                setTimezone(customTimeZoneLabel);
                return;
            }
            setTimezone(Intl.DateTimeFormat('en-US', {
                timeZoneName: 'short'
            }).formatToParts(now).find((part)=>part.type === 'timeZoneName')?.value ?? format(now, 'z'));
        } else {
            setTimezone(undefined);
        }
    }, [
        showTimeZone,
        customTimeZoneLabel
    ]);
    useEffect(()=>{
        if (value) {
            if (value instanceof Date && isValid(value)) {
                setDateAndDateInputValue(value);
            } else {
                if (isValid(new Date(value))) {
                    setDateAndDateInputValue(new Date(value));
                } else {
                    setDateAndDateInputValue(undefined);
                }
            }
        } else {
            setDateAndDateInputValue(undefined);
        }
    }, [
        value,
        setDateAndDateInputValue
    ]);
    const handleChange = useCallback((date, isCalendarUpdate)=>{
        if (onChange) {
            onChange({
                target: {
                    name,
                    value: date
                },
                type: 'change',
                updateLocation: isCalendarUpdate ? 'calendar' : 'input'
            });
        }
    }, [
        onChange,
        name
    ]);
    const handleDatePickerUpdate = (date)=>{
        // Prevent the blur triggering an update event as well
        setPreventBlurUpdate(true);
        setDate((prevDate)=>{
            setDateInputValue(date);
            // Set default time if date is set the first time
            if (!prevDate && date && includeTime && defaultTime) {
                const timeSplit = defaultTime.split(':');
                date?.setHours(+timeSplit[0]);
                date?.setMinutes(+timeSplit[1]);
                date.setSeconds(timeSplit.length > 2 ? +timeSplit[2] : 0);
            } else if (prevDate && date && includeTime) {
                date.setHours(prevDate.getHours());
                date.setMinutes(prevDate.getMinutes());
                if (includeSeconds) {
                    date.setSeconds(prevDate.getSeconds());
                }
            }
            handleChange(date, true);
            return date;
        });
        if (!includeTime) {
            setIsVisible(false);
        }
    };
    const handleInputUpdate = (dateString)=>{
        processInputDate(dateString);
    };
    const handleBlur = (e)=>{
        if (isVisible) {
            return;
        }
        if (preventBlurUpdate) {
            // If the datepicker is opened/closed, prevent the blur update
            setPreventBlurUpdate(false);
            return;
        }
        const newValue = new Date(e.target.value);
        if (newValue && isValid(newValue)) {
            processInputDate(e.target.value, true);
        } else {
            processInputDate(undefined, true);
        }
        restProps?.onBlur?.(e);
    };
    const [minimumAllowedDate, minimumAllowedDateString] = useMemo(()=>{
        return formatDateLimitForInput(min, Boolean(includeTime), Boolean(includeSeconds), DEFAULT_MIN_DATE);
    }, [
        min,
        includeTime,
        includeSeconds
    ]);
    const [maximumAllowedDate, maximumAllowedDateString] = useMemo(()=>{
        return formatDateLimitForInput(max, Boolean(includeTime), Boolean(includeSeconds), DEFAULT_MAX_DATE);
    }, [
        max,
        includeTime,
        includeSeconds
    ]);
    // shouldEmitOnChange should be true for instances when the updated date should propagate to the parent
    // when just typing, shouldEmitOnChange should be false
    // when blurring the input, shouldEmitOnChange should be true
    const processInputDate = (updatedDateString, shouldEmitOnChange = false)=>{
        const updatedDate = updatedDateString ? parseISO(updatedDateString) : undefined;
        // If the input is empty, set the date to undefined and emit the change event if shouldEmitOnChange is true
        if (!updatedDate) {
            setDateAndDateInputValue(undefined);
            if (shouldEmitOnChange) {
                handleChange(undefined, false);
            }
            return;
        }
        // If the date is invalid, don't update the date, the user is still typing
        if (!isValid(updatedDate)) {
            return;
        }
        let validInputDate = updatedDate;
        if (!includeTime) {
            validInputDate = correctDateTimezoneOffset(updatedDate);
        }
        // Don't emit onChange events for dates that have less than 4 digits in the year and before 1900
        // Only allow the input to be updated so the user can keep typing the date
        if (isBefore(validInputDate, new Date(DEFAULT_MIN_DATE)) || isAfter(validInputDate, new Date(DEFAULT_MAX_DATE))) {
            if (shouldEmitOnChange) {
                setDateAndDateInputValue(value);
            } else {
                setDateAndDateInputValue(validInputDate);
            }
            return;
        }
        // If the value should be propagated, it means the input is blurred and the limit checks have already been done
        // so we can update the date and emit the change event
        if (shouldEmitOnChange) {
            if (validInputDate !== value) {
                setDateAndDateInputValue(validInputDate);
                handleChange(validInputDate, false);
            }
            return;
        }
        // Verify limits of the date and don't update the date if it's out of range
        if (minimumAllowedDate && isBefore(validInputDate, minimumAllowedDate)) {
            validInputDate = minimumAllowedDate;
        }
        if (maximumAllowedDate && isAfter(validInputDate, maximumAllowedDate)) {
            validInputDate = maximumAllowedDate;
        }
        setDateAndDateInputValue(validInputDate);
        if (!includeTime && isVisible) {
            setIsVisible(false);
        }
    };
    const handleClear = useCallback(()=>{
        setDateAndDateInputValue(undefined);
        handleChange(undefined, false);
        onClear?.();
    }, [
        onClear,
        handleChange,
        setDateAndDateInputValue
    ]);
    const handleTimeUpdate = (e)=>{
        const newTime = e.nativeEvent?.target?.value;
        const time = date && isValid(date) ? format(date, includeSeconds ? 'HH:mm:ss' : 'HH:mm') : undefined;
        if (newTime && newTime !== time) {
            if (date) {
                const updatedDate = new Date(date);
                const timeSplit = newTime.split(':');
                updatedDate.setHours(+timeSplit[0]);
                updatedDate.setMinutes(+timeSplit[1]);
                if (includeSeconds) {
                    updatedDate.setSeconds(+timeSplit[2]);
                }
                processInputDate(format(updatedDate, format$1));
            }
        }
    };
    // Manually add the clear icon click event listener to avoid reopening the datepicker when clearing the input
    useEffect(()=>{
        if (allowClear && inputRef.current && !useNewDatePickerInput) {
            const clearIcon = inputRef.current.input?.closest('[type="button"]')?.querySelector(`.${classNamePrefix}-input-clear-icon`);
            if (clearIcon !== fromClearRef.current) {
                fromClearRef.current = clearIcon;
                const clientEventListener = (e)=>{
                    e.stopPropagation();
                    e.preventDefault();
                    handleClear();
                };
                clearIcon.addEventListener('click', clientEventListener);
            }
        }
    }, [
        classNamePrefix,
        defaultTime,
        handleClear,
        allowClear,
        useNewDatePickerInput
    ]);
    const { classNames, datePickerStyles } = useMemo(()=>({
            classNames: generateDatePickerClassNames(`${classNamePrefix}-datepicker`),
            datePickerStyles: getDayPickerStyles(`${classNamePrefix}-datepicker`, theme)
        }), [
        classNamePrefix,
        theme
    ]);
    const chevronLeftIconComp = (props)=>/*#__PURE__*/ jsx(ChevronLeftIcon, {
            ...props
        });
    const chevronRightIconComp = (props)=>/*#__PURE__*/ jsx(ChevronRightIcon, {
            ...props
        });
    return /*#__PURE__*/ jsx("div", {
        className: `${classNamePrefix}-datepicker`,
        css: {
            width,
            minWidth,
            maxWidth,
            pointerEvents: restProps?.disabled ? 'none' : 'auto'
        },
        ...wrapperProps,
        children: /*#__PURE__*/ jsxs(Root$3, {
            componentId: "codegen_design-system_src_development_datepicker_datepicker.tsx_330",
            open: isVisible,
            onOpenChange: setIsVisible,
            children: [
                /*#__PURE__*/ jsx(Trigger, {
                    asChild: true,
                    disabled: restProps?.disabled,
                    role: "combobox",
                    children: /*#__PURE__*/ jsx("div", {
                        children: useNewDatePickerInput ? /*#__PURE__*/ jsxs(Fragment, {
                            children: [
                                /*#__PURE__*/ jsx(DatePickerInput, {
                                    id: id,
                                    // Once we get rid of the old datepicker input, we can remove the as any
                                    ref: inputRef,
                                    name: name,
                                    validationState: validationState,
                                    allowClear: allowClear,
                                    onClear: allowClear ? handleClear : undefined,
                                    placeholder: "Select Date",
                                    "aria-label": includeTime ? 'Select Date and Time' : 'Select Date',
                                    prefix: "Date:",
                                    role: "textbox",
                                    suffix: showTimeZone ? /*#__PURE__*/ jsx("span", {
                                        children: timezone
                                    }) : undefined,
                                    min: minimumAllowedDateString,
                                    max: maximumAllowedDateString,
                                    ...restProps,
                                    type: includeTime ? 'datetime-local' : 'date',
                                    step: includeTime && includeSeconds ? 1 : undefined,
                                    onKeyDown: (event)=>handleInputKeyDown(event, setIsVisible, ()=>handleInputUpdate(undefined)),
                                    onChange: (e)=>handleInputUpdate(e.target.value),
                                    value: date && isValid(date) ? format(date, format$1) : undefined,
                                    onBlur: handleBlur,
                                    onClick: (e)=>{
                                        e.preventDefault();
                                        e.stopPropagation();
                                        setIsVisible(!isVisible);
                                        restProps?.onClick?.(e);
                                    }
                                }),
                                /*#__PURE__*/ jsx("input", {
                                    type: "hidden",
                                    ref: ref,
                                    value: date || ''
                                })
                            ]
                        }) : /*#__PURE__*/ jsxs(Fragment, {
                            children: [
                                /*#__PURE__*/ jsx(Input, {
                                    id: id,
                                    ref: inputRef,
                                    name: name,
                                    validationState: validationState,
                                    allowClear: allowClear,
                                    placeholder: "Select Date",
                                    "aria-label": includeTime ? 'Select Date and Time' : 'Select Date',
                                    prefix: "Date:",
                                    role: "textbox",
                                    suffix: showTimeZone ? /*#__PURE__*/ jsx("span", {
                                        children: timezone
                                    }) : undefined,
                                    min: minimumAllowedDateString,
                                    max: maximumAllowedDateString,
                                    ...restProps,
                                    css: {
                                        // Chrome, safari & webkit browsers specific fix to hide the calendar picker indicator
                                        '*::-webkit-calendar-picker-indicator': {
                                            display: 'none'
                                        },
                                        [`.${classNamePrefix}-input`]: {
                                            // vertical alignment fix for all browsers except chrome
                                            display: 'inline-flex',
                                            // Firefox specific fix to hide the calendar picker indicator
                                            marginRight: '-32px',
                                            paddingRight: '32px !important',
                                            width: 'calc(100% + 32px)',
                                            clipPath: 'inset(0 32px 0 0)'
                                        },
                                        [`.${classNamePrefix}-input-prefix`]: {
                                            ...!restProps?.disabled && {
                                                color: `${theme.colors.textPrimary} !important`
                                            }
                                        },
                                        [`&.${classNamePrefix}-input-affix-wrapper > *`]: {
                                            height: theme.typography.lineHeightBase
                                        },
                                        ...showTimeZone && {
                                            [`.${classNamePrefix}-input-suffix`]: {
                                                display: 'inline-flex',
                                                flexDirection: 'row-reverse',
                                                gap: theme.spacing.sm,
                                                alignItems: 'center'
                                            }
                                        }
                                    },
                                    type: includeTime ? 'datetime-local' : 'date',
                                    step: includeTime && includeSeconds ? 1 : undefined,
                                    onKeyDown: (event)=>handleInputKeyDown(event, setIsVisible, ()=>handleInputUpdate(undefined)),
                                    onChange: (e)=>handleInputUpdate(e.target.value),
                                    value: date && isValid(date) ? format(date, format$1) : undefined,
                                    onBlur: handleBlur,
                                    onClick: (e)=>{
                                        e.preventDefault();
                                        e.stopPropagation();
                                        setIsVisible(!isVisible);
                                        restProps?.onClick?.(e);
                                    }
                                }),
                                /*#__PURE__*/ jsx("input", {
                                    type: "hidden",
                                    ref: ref,
                                    value: date || ''
                                })
                            ]
                        })
                    })
                }),
                /*#__PURE__*/ jsxs(Content, {
                    align: "start",
                    css: datePickerStyles,
                    children: [
                        /*#__PURE__*/ jsx(DayPicker, {
                            initialFocus: true,
                            ...datePickerProps,
                            mode: mode,
                            selected: mode === 'range' ? selected : date,
                            onDayClick: handleDatePickerUpdate,
                            showOutsideDays: mode === 'range' ? false : true,
                            formatters: {
                                formatWeekdayName: (date)=>format(date, 'iiiii', {
                                        locale: datePickerProps?.locale
                                    })
                            },
                            components: {
                                Day,
                                IconLeft: chevronLeftIconComp,
                                IconRight: chevronRightIconComp
                            },
                            defaultMonth: date,
                            classNames: classNames
                        }),
                        quickActions?.length && /*#__PURE__*/ jsx("div", {
                            style: {
                                display: 'flex',
                                gap: theme.spacing.sm,
                                marginBottom: theme.spacing.md,
                                padding: `${theme.spacing.xs}px ${theme.spacing.xs}px 0`,
                                maxWidth: 225,
                                flexWrap: 'wrap'
                            },
                            children: quickActions?.map((action, i)=>/*#__PURE__*/ jsx(Button$1, {
                                    size: "small",
                                    componentId: "codegen_design-system_src_design-system_datepicker_datepicker.tsx_281",
                                    onClick: ()=>action.onClick ? action.onClick(action.value) : !Array.isArray(action.value) && handleDatePickerUpdate(action.value),
                                    children: action.label
                                }, i))
                        }),
                        includeTime && /*#__PURE__*/ jsx(Input, {
                            componentId: "codegen_design-system_src_development_datepicker_datepicker.tsx_306",
                            type: "time",
                            step: includeSeconds ? 1 : undefined,
                            "aria-label": "Time",
                            role: "textbox",
                            min: minimumAllowedDateString,
                            max: maximumAllowedDateString,
                            ...timeInputProps,
                            value: date && isValid(date) ? format(date, includeSeconds ? 'HH:mm:ss' : 'HH:mm') : undefined,
                            onChange: handleTimeUpdate,
                            css: {
                                '*::-webkit-calendar-picker-indicator': {
                                    position: 'absolute',
                                    right: -8,
                                    width: theme.general.iconSize,
                                    height: theme.general.iconSize,
                                    zIndex: theme.options.zIndexBase + 1,
                                    color: 'transparent',
                                    background: 'transparent'
                                },
                                [`.${classNamePrefix}-input-suffix`]: {
                                    position: 'absolute',
                                    right: 12,
                                    top: 8
                                }
                            },
                            suffix: /*#__PURE__*/ jsx(ClockIcon, {}),
                            disabled: timeInputProps?.disabled
                        }),
                        mode === 'range' && includeTime && onOkPress && /*#__PURE__*/ jsx("div", {
                            css: {
                                paddingTop: theme.spacing.md,
                                display: 'flex',
                                justifyContent: 'flex-end'
                            },
                            children: /*#__PURE__*/ jsx(Button$1, {
                                "aria-label": "Open end date picker",
                                type: "primary",
                                componentId: "datepicker-dubois-ok-button",
                                onClick: ()=>{
                                    // Commit current input value immediately to parent before closing
                                    handleChange(date, false);
                                    onOkPress?.();
                                },
                                children: okButtonLabel ?? 'Ok'
                            })
                        })
                    ]
                })
            ]
        })
    });
});
const getRangeQuickActionsBasic = ({ today, yesterday, lastWeek })=>{
    const todayStart = startOfToday();
    const weekStart = startOfWeek(todayStart);
    return [
        {
            label: 'Today',
            value: [
                todayStart,
                endOfToday()
            ],
            ...today
        },
        {
            label: 'Yesterday',
            value: [
                startOfYesterday(),
                endOfYesterday()
            ],
            ...yesterday
        },
        {
            label: 'Last week',
            value: [
                sub(weekStart, {
                    days: 7
                }),
                sub(weekStart, {
                    days: 1
                })
            ],
            ...lastWeek
        }
    ];
};
const RangePicker = (props)=>{
    const { id, onChange, startDatePickerProps, endDatePickerProps, includeTime, includeSeconds, allowClear, minWidth, maxWidth, width, disabled, quickActions, wrapperProps, noRangeValidation } = props;
    const [range, setRange] = useState({
        from: startDatePickerProps?.value,
        to: endDatePickerProps?.value
    });
    const { classNamePrefix } = useDesignSystemTheme();
    // Focus is lost when the popover is closed, we need to set the focus back to the input that opened the popover manually.
    const [isFromVisible, setIsFromVisible] = useState(false);
    const [isToVisible, setIsToVisible] = useState(false);
    const [isRangeInputFocused, setIsRangeInputFocused] = useState(false);
    const fromInputRef = useRef(null);
    const toInputRef = useRef(null);
    useImperativeHandle(startDatePickerProps?.ref, ()=>fromInputRef.current);
    useImperativeHandle(endDatePickerProps?.ref, ()=>toInputRef.current);
    const fromInputRefVisible = useRef(isFromVisible);
    const toInputRefVisible = useRef(isToVisible);
    useEffect(()=>{
        if (!isFromVisible && fromInputRefVisible.current) {
            fromInputRef.current?.focus();
        }
        fromInputRefVisible.current = isFromVisible;
    }, [
        isFromVisible
    ]);
    useEffect(()=>{
        if (!isToVisible && toInputRefVisible.current) {
            toInputRef.current?.focus();
        }
        toInputRefVisible.current = isToVisible;
    }, [
        isToVisible
    ]);
    const checkIfDateTimeIsDisabled = useCallback((date, isStart = false)=>{
        const dateToCompareTo = isStart ? range?.to : range?.from;
        if (date && dateToCompareTo) {
            return isStart ? isAfter(date, dateToCompareTo) : isBefore(date, dateToCompareTo);
        }
        return false;
    }, [
        range
    ]);
    useEffect(()=>{
        setRange((prevValue)=>({
                from: startDatePickerProps?.value,
                to: prevValue?.to
            }));
    }, [
        startDatePickerProps?.value
    ]);
    useEffect(()=>{
        setRange((prevValue)=>({
                from: prevValue?.from,
                to: endDatePickerProps?.value
            }));
    }, [
        endDatePickerProps?.value
    ]);
    const quickActionsWithHandler = useMemo(()=>{
        if (quickActions) {
            return quickActions.map((action)=>{
                if (Array.isArray(action.value)) {
                    return {
                        ...action,
                        onClick: (value)=>{
                            setRange({
                                from: value[0],
                                to: value[1]
                            });
                            onChange?.({
                                target: {
                                    name: props.name,
                                    value: {
                                        from: value[0],
                                        to: value[1]
                                    }
                                },
                                type: 'change',
                                updateLocation: 'preset'
                            });
                            action.onClick?.(value);
                            setIsFromVisible(false);
                            setIsToVisible(false);
                        }
                    };
                }
                return action;
            });
        }
        return quickActions;
    }, [
        quickActions,
        onChange,
        props.name
    ]);
    const handleUpdateDate = useCallback((e, isStart)=>{
        const date = e.target.value;
        const newRange = isStart ? {
            from: date,
            to: range?.to
        } : {
            from: range?.from,
            to: date
        };
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
        onChange?.({
            target: {
                name: props.name,
                value: newRange
            },
            type: 'change',
            updateLocation: e.updateLocation
        });
    }, [
        onChange,
        includeTime,
        startDatePickerProps,
        endDatePickerProps,
        range,
        props.name
    ]);
    // Use useMemo to calculate disabled dates
    const disabledDates = useMemo(()=>{
        let startDisabledFromProps, endDisabledFromProps;
        if (startDatePickerProps?.datePickerProps?.disabled) {
            startDisabledFromProps = Array.isArray(startDatePickerProps?.datePickerProps?.disabled) ? startDatePickerProps?.datePickerProps?.disabled : [
                startDatePickerProps?.datePickerProps?.disabled
            ];
        }
        const startDisabled = [
            {
                after: range?.to
            },
            ...startDisabledFromProps ? startDisabledFromProps : []
        ].filter(Boolean);
        if (endDatePickerProps?.datePickerProps?.disabled) {
            endDisabledFromProps = Array.isArray(endDatePickerProps?.datePickerProps?.disabled) ? endDatePickerProps?.datePickerProps?.disabled : [
                endDatePickerProps?.datePickerProps?.disabled
            ];
        }
        const endDisabled = [
            {
                before: range?.from
            },
            ...endDisabledFromProps ? endDisabledFromProps : []
        ].filter(Boolean);
        return {
            startDisabled,
            endDisabled
        };
    }, [
        range?.from,
        range?.to,
        startDatePickerProps?.datePickerProps?.disabled,
        endDatePickerProps?.datePickerProps?.disabled
    ]);
    const [startDateMaxInputDate] = useMemo(()=>formatDateLimitForInput(noRangeValidation ? undefined : range?.to, Boolean(includeTime), Boolean(includeSeconds), DEFAULT_MAX_DATE), [
        includeTime,
        includeSeconds,
        noRangeValidation,
        range?.to
    ]);
    const [endDateMinInputDate] = useMemo(()=>formatDateLimitForInput(noRangeValidation ? undefined : range?.from, Boolean(includeTime), Boolean(includeSeconds), DEFAULT_MIN_DATE), [
        includeTime,
        includeSeconds,
        noRangeValidation,
        range?.from
    ]);
    const openEndDatePicker = ()=>{
        setIsFromVisible(false);
        setIsToVisible(true);
    };
    const closeEndDatePicker = ()=>{
        setIsToVisible(false);
    };
    const handleTimePickerKeyPress = (e)=>{
        if (e.key === 'Enter') {
            openEndDatePicker();
        }
        props.startDatePickerProps?.timeInputProps?.onKeyDown?.(e);
    };
    return /*#__PURE__*/ jsxs("div", {
        className: `${classNamePrefix}-rangepicker`,
        ...wrapperProps,
        "data-focused": isRangeInputFocused,
        css: {
            display: 'flex',
            alignItems: 'center',
            minWidth,
            maxWidth,
            width
        },
        children: [
            /*#__PURE__*/ jsx(DatePicker, {
                quickActions: quickActionsWithHandler,
                prefix: "Start:",
                open: isFromVisible,
                onOpenChange: setIsFromVisible,
                okButtonLabel: "Next",
                ...startDatePickerProps,
                id: id,
                ref: fromInputRef,
                disabled: disabled || startDatePickerProps?.disabled,
                onChange: (e)=>handleUpdateDate(e, true),
                includeTime: includeTime,
                includeSeconds: includeSeconds,
                allowClear: allowClear,
                max: startDateMaxInputDate,
                datePickerProps: {
                    ...startDatePickerProps?.datePickerProps,
                    disabled: disabledDates.startDisabled
                },
                timeInputProps: {
                    onKeyDown: handleTimePickerKeyPress
                },
                // @ts-expect-error - DatePickerProps does not have a mode property in the public API but is needed for this use case
                mode: "range",
                selected: range,
                value: range?.from,
                dateTimeDisabledFn: (date)=>checkIfDateTimeIsDisabled(date, true),
                onFocus: (e)=>{
                    setIsRangeInputFocused(true);
                    startDatePickerProps?.onFocus?.(e);
                },
                onBlur: (e)=>{
                    setIsRangeInputFocused(false);
                    startDatePickerProps?.onBlur?.(e);
                },
                css: {
                    '*::-webkit-calendar-picker-indicator': {
                        display: 'none'
                    },
                    borderTopRightRadius: 0,
                    borderBottomRightRadius: 0
                },
                wrapperProps: {
                    style: {
                        width: '50%'
                    }
                },
                onOkPress: openEndDatePicker
            }),
            /*#__PURE__*/ jsx(DatePicker, {
                quickActions: quickActionsWithHandler,
                prefix: "End:",
                min: endDateMinInputDate,
                okButtonLabel: "Close",
                ...endDatePickerProps,
                ref: toInputRef,
                disabled: disabled || endDatePickerProps?.disabled,
                onChange: (e)=>handleUpdateDate(e, false),
                includeTime: includeTime,
                includeSeconds: includeSeconds,
                open: isToVisible,
                onOpenChange: setIsToVisible,
                allowClear: allowClear,
                datePickerProps: {
                    ...endDatePickerProps?.datePickerProps,
                    disabled: disabledDates.endDisabled
                },
                // @ts-expect-error - DatePickerProps does not have a mode property in the public API but is needed for this use case
                mode: "range",
                selected: range,
                value: range?.to,
                dateTimeDisabledFn: (date)=>checkIfDateTimeIsDisabled(date, false),
                onFocus: (e)=>{
                    setIsRangeInputFocused(true);
                    startDatePickerProps?.onFocus?.(e);
                },
                onBlur: (e)=>{
                    setIsRangeInputFocused(false);
                    startDatePickerProps?.onBlur?.(e);
                },
                css: {
                    borderTopLeftRadius: 0,
                    borderBottomLeftRadius: 0,
                    left: -1
                },
                wrapperProps: {
                    style: {
                        width: '50%'
                    }
                },
                onOkPress: closeEndDatePicker
            })
        ]
    });
};

const ListboxContext = /*#__PURE__*/ createContext(null);
const useListboxContext = ()=>{
    const context = useContext(ListboxContext);
    if (!context) {
        throw new Error('useListboxContext must be used within a ListboxProvider');
    }
    return context;
};
const ListboxRoot = ({ children, className, onSelect, initialSelectedValue, listBoxDivRef })=>{
    const [selectedValue, setSelectedValue] = useState(initialSelectedValue);
    const [highlightedValue, setHighlightedValue] = useState();
    // Generate stable unique ID on first render (better for SSR than Math.random())
    const idRef = useRef();
    if (!idRef.current) {
        idRef.current = `listbox-${generateUuidV4()}`;
    }
    const listboxId = idRef.current;
    const getContentOptions = (element)=>{
        const options = element?.querySelectorAll('[role="option"], [role="link"]');
        return options ? Array.from(options) : undefined;
    };
    const handleKeyNavigation = useCallback((event, options)=>{
        const currentIndex = options.findIndex((option)=>option.value === highlightedValue);
        let nextIndex = currentIndex;
        switch(event.key){
            case 'ArrowDown':
                event.preventDefault();
                nextIndex = currentIndex < options.length - 1 ? currentIndex + 1 : 0;
                break;
            case 'ArrowUp':
                event.preventDefault();
                nextIndex = currentIndex > 0 ? currentIndex - 1 : options.length - 1;
                break;
            case 'Home':
                event.preventDefault();
                nextIndex = 0;
                break;
            case 'End':
                event.preventDefault();
                nextIndex = options.length - 1;
                break;
            case 'Enter':
            case ' ':
                event.preventDefault();
                if (highlightedValue !== undefined) {
                    onSelect?.(highlightedValue);
                    if (options[currentIndex].href) {
                        window.open(options[currentIndex].href, '_blank');
                    } else {
                        setSelectedValue(highlightedValue);
                    }
                }
                break;
            default:
                return;
        }
        if (nextIndex !== currentIndex && listBoxDivRef?.current) {
            setHighlightedValue(options[nextIndex].value);
            const optionsList = getContentOptions(listBoxDivRef?.current);
            if (optionsList && optionsList[nextIndex]) {
                const nextOption = optionsList[nextIndex];
                nextOption.scrollIntoView?.({
                    block: 'center'
                });
                // Update aria-activedescendant immediately for screen reader announcement.
                // This is imperative DOM manipulation to ensure the screen reader announces
                // the highlighted option instantly during keyboard navigation, before React re-renders.
                // Matches DialogCombobox's highlightOption function pattern.
                const input = listBoxDivRef.current.querySelector('input[type="text"], input[type="search"]');
                const listbox = listBoxDivRef.current.querySelector('[role="listbox"]');
                const activeElement = input || listbox;
                if (activeElement && nextOption.id) {
                    activeElement.setAttribute('aria-activedescendant', nextOption.id);
                }
            }
        }
    }, [
        highlightedValue,
        onSelect,
        listBoxDivRef
    ]);
    const contextValue = useMemo(()=>({
            selectedValue,
            setSelectedValue,
            highlightedValue,
            setHighlightedValue,
            listboxId,
            handleKeyNavigation
        }), [
        selectedValue,
        highlightedValue,
        listboxId,
        handleKeyNavigation
    ]);
    return /*#__PURE__*/ jsx(ListboxContext.Provider, {
        value: contextValue,
        children: /*#__PURE__*/ jsx("div", {
            className: className,
            children: children
        })
    });
};

const ListboxInput = ({ value, onChange, placeholder, 'aria-controls': ariaControls, 'aria-activedescendant': ariaActiveDescendant, className, options })=>{
    const { handleKeyNavigation } = useListboxContext();
    const designSystemTheme = useDesignSystemTheme();
    const handleChange = useCallback((event)=>{
        onChange(event.target.value);
    }, [
        onChange
    ]);
    const handleKeyDown = useCallback((event)=>{
        // Only handle navigation keys if there are options
        if (options.length > 0 && [
            'ArrowDown',
            'ArrowUp',
            'Home',
            'End',
            'Enter'
        ].includes(event.key)) {
            handleKeyNavigation(event, options);
        }
    }, [
        handleKeyNavigation,
        options
    ]);
    return /*#__PURE__*/ jsx("div", {
        css: {
            position: 'sticky',
            top: 0,
            background: designSystemTheme.theme.colors.backgroundPrimary,
            zIndex: designSystemTheme.theme.options.zIndexBase + 1
        },
        children: /*#__PURE__*/ jsx(Input, {
            componentId: "listbox-filter-input",
            role: "combobox",
            "aria-controls": ariaControls,
            "aria-activedescendant": ariaActiveDescendant,
            "aria-expanded": "true",
            "aria-autocomplete": "list",
            value: value,
            onChange: handleChange,
            onKeyDown: handleKeyDown,
            placeholder: placeholder,
            prefix: /*#__PURE__*/ jsx(SearchIcon, {}),
            className: className,
            allowClear: true
        })
    });
};

const ListboxOptions = ({ options, onSelect, onHighlight, className })=>{
    const theme = useTheme();
    const { listboxId, selectedValue, setSelectedValue, highlightedValue, handleKeyNavigation } = useListboxContext();
    const listboxRef = useRef(null);
    const handleKeyDown = useCallback((event)=>{
        handleKeyNavigation(event, options);
    }, [
        handleKeyNavigation,
        options
    ]);
    const handleClick = useCallback((event, option)=>{
        onSelect?.(option.value);
        if (option.href) {
            event.preventDefault();
            window.open(option.href, '_blank');
        } else {
            setSelectedValue(option.value);
        }
    }, [
        setSelectedValue,
        onSelect
    ]);
    useEffect(()=>{
        // If no option is highlighted, highlight the first one
        if (!highlightedValue && options.length > 0) {
            onHighlight(options[0].value);
        }
    }, [
        highlightedValue,
        onHighlight,
        options
    ]);
    return /*#__PURE__*/ jsx("div", {
        ref: listboxRef,
        role: "listbox",
        id: listboxId,
        className: className,
        tabIndex: 0,
        onKeyDown: handleKeyDown,
        "aria-activedescendant": highlightedValue ? `${listboxId}-${highlightedValue}` : undefined,
        css: /*#__PURE__*/ css({
            outline: 'none',
            '&:focus-visible': {
                boxShadow: `0 0 0 2px ${theme.colors.actionDefaultBorderFocus}`,
                borderRadius: theme.borders.borderRadiusSm
            }
        }),
        children: options.map((option)=>(option.renderOption || ((additionalProps)=>/*#__PURE__*/ jsx("div", {
                    ...additionalProps,
                    children: option.label
                })))({
                key: option.value,
                role: option.href ? 'link' : 'option',
                id: `${listboxId}-${option.value}`,
                'aria-selected': option.value === selectedValue,
                onClick: (event)=>handleClick(event, option),
                onMouseEnter: ()=>onHighlight(option.value),
                'data-highlighted': option.value === highlightedValue,
                css: [
                    getComboboxOptionItemWrapperStyles(theme),
                    {
                        '&:focus': {
                            outline: 'none'
                        }
                    }
                ],
                href: option.href,
                tabIndex: -1
            }))
    });
};

const ListboxContent = ({ options, filterValue, setFilterValue, filterInputPlaceholder, onSelect, ariaLabel, includeFilterInput, filterInputEmptyMessage, listBoxDivRef })=>{
    const [highlightedValue, setHighlightedValue] = useState();
    const { listboxId } = useListboxContext();
    const designSystemTheme = useDesignSystemTheme();
    const noResultsId = useMemo(()=>`${listboxId}-no-results`, [
        listboxId
    ]);
    const filteredOptions = useMemo(()=>{
        if (!filterValue) return options;
        const lowerFilter = filterValue.toLowerCase();
        return options.filter((option)=>option.value.toLowerCase().includes(lowerFilter) || option.label.toLowerCase().includes(lowerFilter));
    }, [
        filterValue,
        options
    ]);
    // Update aria-activedescendant to announce results/no-results to screen readers.
    // Using direct DOM manipulation (imperative) rather than React props because:
    // 1. Ensures immediate updates that screen readers can detect during keyboard navigation
    // 2. Provides single source of truth (this + ListboxRoot keyboard handler)
    // 3. Matches DialogCombobox pattern for reliable accessibility
    React__default.useEffect(()=>{
        if (!listBoxDivRef?.current) {
            return;
        }
        // Try to find input first (if filter is enabled), fallback to listbox
        const input = listBoxDivRef.current.querySelector('input[type="text"], input[type="search"]');
        const listbox = listBoxDivRef.current.querySelector('[role="listbox"]');
        const activeElement = input || listbox;
        if (!activeElement) {
            return;
        }
        if (filteredOptions.length === 0) {
            // Point to the no results div so screen readers announce it
            activeElement.setAttribute('aria-activedescendant', noResultsId);
        } else if (highlightedValue) {
            // Point to the highlighted option (when navigating)
            activeElement.setAttribute('aria-activedescendant', `${listboxId}-${highlightedValue}`);
        } else {
            // Clear when no highlight and results exist
            activeElement.removeAttribute('aria-activedescendant');
        }
    }, [
        filteredOptions.length,
        highlightedValue,
        listboxId,
        noResultsId,
        listBoxDivRef
    ]);
    return /*#__PURE__*/ jsxs("div", {
        css: /*#__PURE__*/ css({
            display: 'flex',
            flexDirection: 'column',
            gap: '8px'
        }),
        ref: listBoxDivRef,
        children: [
            includeFilterInput && /*#__PURE__*/ jsx(ListboxInput, {
                value: filterValue,
                onChange: setFilterValue,
                placeholder: filterInputPlaceholder,
                "aria-controls": listboxId,
                options: filteredOptions
            }),
            filteredOptions.length > 0 ? /*#__PURE__*/ jsx("div", {
                "aria-live": "polite",
                css: {
                    width: '100%'
                },
                children: /*#__PURE__*/ jsx(ListboxOptions, {
                    options: filteredOptions,
                    onSelect: onSelect,
                    onHighlight: setHighlightedValue,
                    "aria-label": ariaLabel
                })
            }) : /*#__PURE__*/ jsx("div", {
                id: noResultsId,
                css: {
                    color: designSystemTheme.theme.colors.textSecondary,
                    textAlign: 'center',
                    padding: '6px 12px',
                    width: '100%',
                    boxSizing: 'border-box'
                },
                children: filterInputEmptyMessage ?? 'No results found'
            })
        ]
    });
};
const Listbox = ({ options, onSelect, includeFilterInput, filterInputEmptyMessage, initialSelectedValue, filterInputPlaceholder, 'aria-label': ariaLabel, componentId, analyticsEvents = [
    DesignSystemEventProviderAnalyticsEventTypes.OnValueChange
], valueHasNoPii, className })=>{
    const [filterValue, setFilterValue] = useState('');
    const memoizedAnalyticsEvents = useMemo(()=>analyticsEvents, [
        analyticsEvents
    ]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.Listbox,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        valueHasNoPii
    });
    const handleSelect = useCallback((value)=>{
        eventContext.onValueChange(value);
        onSelect?.(value);
    }, [
        eventContext,
        onSelect
    ]);
    const { elementRef: listBoxDivRef } = useNotifyOnFirstView({
        onView: eventContext.onView,
        value: initialSelectedValue
    });
    return /*#__PURE__*/ jsx(ListboxRoot, {
        className: className,
        onSelect: handleSelect,
        initialSelectedValue: initialSelectedValue,
        listBoxDivRef: listBoxDivRef,
        children: /*#__PURE__*/ jsx(ListboxContent, {
            options: options,
            filterValue: filterValue,
            setFilterValue: setFilterValue,
            filterInputPlaceholder: filterInputPlaceholder,
            onSelect: handleSelect,
            ariaLabel: ariaLabel,
            includeFilterInput: includeFilterInput,
            filterInputEmptyMessage: filterInputEmptyMessage,
            listBoxDivRef: listBoxDivRef
        })
    });
};

const RadioGroupContext = /*#__PURE__*/ React__default.createContext('medium');
const Root$2 = /*#__PURE__*/ React__default.forwardRef(({ size, componentId, analyticsEvents = [
    DesignSystemEventProviderAnalyticsEventTypes.OnValueChange
], valueHasNoPii, onValueChange, ...props }, forwardedRef)=>{
    const { theme } = useDesignSystemTheme();
    const contextValue = React__default.useMemo(()=>size ?? 'medium', [
        size
    ]);
    const memoizedAnalyticsEvents = useMemo(()=>analyticsEvents, [
        analyticsEvents
    ]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.PillControl,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        valueHasNoPii
    });
    const { elementRef: pillControlRef } = useNotifyOnFirstView({
        onView: eventContext.onView,
        value: props.value ?? props.defaultValue
    });
    const onValueChangeWrapper = useCallback((value)=>{
        eventContext.onValueChange?.(value);
        onValueChange?.(value);
    }, [
        eventContext,
        onValueChange
    ]);
    const mergedRef = useMergeRefs([
        forwardedRef,
        pillControlRef
    ]);
    return /*#__PURE__*/ jsx(RadioGroupContext.Provider, {
        value: contextValue,
        children: /*#__PURE__*/ jsx(RadioGroup, {
            css: {
                display: 'flex',
                flexWrap: 'wrap',
                gap: theme.spacing.sm
            },
            onValueChange: onValueChangeWrapper,
            ...props,
            ref: mergedRef,
            ...eventContext.dataComponentProps
        })
    });
});
const Item = /*#__PURE__*/ React__default.forwardRef(({ children, icon, ...props }, forwardedRef)=>{
    const size = React__default.useContext(RadioGroupContext);
    const { theme } = useDesignSystemTheme();
    const iconClass = 'pill-control-icon';
    const css = useRadioGroupItemStyles(size, iconClass);
    return /*#__PURE__*/ jsxs(RadioGroupItem, {
        css: css,
        ...props,
        children: [
            icon && /*#__PURE__*/ jsx("span", {
                className: iconClass,
                css: {
                    marginRight: size === 'large' ? theme.spacing.sm : theme.spacing.xs,
                    [`& > .anticon`]: {
                        verticalAlign: `-3px`
                    }
                },
                children: icon
            }),
            children
        ]
    });
});
const useRadioGroupItemStyles = (size, iconClass)=>{
    const { theme } = useDesignSystemTheme();
    const useNewBorderColors = safex('databricks.fe.designsystem.useNewBorderColors', false);
    return {
        textOverflow: 'ellipsis',
        boxShadow: theme.shadows.xs,
        whiteSpace: 'nowrap',
        overflow: 'hidden',
        appearance: 'none',
        textDecoration: 'none',
        background: 'none',
        border: '1px solid',
        cursor: 'pointer',
        backgroundColor: theme.colors.actionDefaultBackgroundDefault,
        borderColor: useNewBorderColors ? theme.colors.actionDefaultBorderDefault : theme.colors.border,
        color: theme.colors.textPrimary,
        lineHeight: theme.typography.lineHeightBase,
        height: 32,
        paddingInline: '12px',
        fontWeight: theme.typography.typographyRegularFontWeight,
        fontSize: theme.typography.fontSizeBase,
        borderRadius: theme.borders.borderRadiusFull,
        transition: 'background-color 0.2s ease-in-out, border-color 0.2s ease-in-out',
        [`& > .${iconClass}`]: {
            color: theme.colors.textSecondary,
            ...size === 'large' ? {
                backgroundColor: theme.colors.tagDefault,
                padding: theme.spacing.sm,
                borderRadius: theme.borders.borderRadiusFull
            } : {}
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
                color: 'inherit'
            },
            [`& > .${iconClass}, &:hover > .${iconClass}`]: {
                color: theme.colors.actionDefaultTextPress,
                ...size === 'large' ? {
                    backgroundColor: theme.colors.actionIconBackgroundPress
                } : {}
            }
        },
        '&:focus-visible': {
            outlineStyle: 'solid',
            outlineWidth: '2px',
            outlineOffset: '0px',
            outlineColor: theme.colors.actionDefaultBorderFocus
        },
        '&:hover': {
            backgroundColor: theme.colors.actionDefaultBackgroundHover,
            borderColor: theme.colors.actionLinkHover,
            color: theme.colors.actionDefaultTextHover,
            [`& > .${iconClass}`]: {
                color: 'inherit',
                ...size === 'large' ? {
                    backgroundColor: theme.colors.actionIconBackgroundHover
                } : {}
            }
        },
        '&:active': {
            backgroundColor: theme.colors.actionDefaultBackgroundPress,
            borderColor: theme.colors.actionLinkPress,
            color: theme.colors.actionDefaultTextPress,
            [`& > .${iconClass}`]: {
                color: 'inherit',
                ...size === 'large' ? {
                    backgroundColor: theme.colors.actionIconBackgroundPress
                } : {}
            }
        },
        '&:disabled': {
            backgroundColor: theme.colors.actionDisabledBackground,
            borderColor: theme.colors.actionDisabledBorder,
            color: theme.colors.actionDisabledText,
            cursor: 'not-allowed',
            [`& > .${iconClass}`]: {
                color: 'inherit'
            }
        },
        ...size === 'small' ? {
            height: 24,
            lineHeight: theme.typography.lineHeightSm,
            paddingInline: theme.spacing.sm
        } : {},
        ...size === 'large' ? {
            height: 44,
            lineHeight: theme.typography.lineHeightXl,
            paddingInline: theme.spacing.md,
            paddingInlineStart: '6px',
            borderRadius: theme.borders.borderRadiusFull
        } : {}
    };
};

var PillControl = /*#__PURE__*/Object.freeze({
  __proto__: null,
  Item: Item,
  Root: Root$2
});

const progressContextDefaults = {
    progress: 0
};
const ProgressContext = /*#__PURE__*/ createContext(progressContextDefaults);
const ProgressContextProvider = ({ children, value })=>{
    return /*#__PURE__*/ jsx(ProgressContext.Provider, {
        value: value,
        children: children
    });
};

const getProgressRootStyles = (theme, { minWidth, maxWidth })=>{
    const styles = {
        position: 'relative',
        overflow: 'hidden',
        backgroundColor: theme.colors.progressTrack,
        height: theme.spacing.sm,
        width: '100%',
        borderRadius: theme.borders.borderRadiusFull,
        ...minWidth && {
            minWidth
        },
        ...maxWidth && {
            maxWidth
        },
        /* Fix overflow clipping in Safari */ /* https://gist.github.com/domske/b66047671c780a238b51c51ffde8d3a0 */ transform: 'translateZ(0)'
    };
    return /*#__PURE__*/ css(importantify(styles));
};
const Root$1 = (props)=>{
    const { children, value, minWidth, maxWidth, ...restProps } = props;
    const { theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsx(ProgressContextProvider, {
        value: {
            progress: value
        },
        children: /*#__PURE__*/ jsx(Progress$1.Root, {
            value: value,
            ...restProps,
            css: getProgressRootStyles(theme, {
                minWidth,
                maxWidth
            }),
            children: children
        })
    });
};
const getProgressIndicatorStyles = (theme)=>{
    const styles = {
        backgroundColor: theme.colors.progressFill,
        height: '100%',
        width: '100%',
        transition: 'transform 300ms linear',
        borderRadius: theme.borders.borderRadiusFull
    };
    return /*#__PURE__*/ css(importantify(styles));
};
const Indicator = (props)=>{
    const { progress } = React__default.useContext(ProgressContext);
    const { theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsx(Progress$1.Indicator, {
        css: getProgressIndicatorStyles(theme),
        style: {
            transform: `translateX(-${100 - (progress ?? 100)}%)`
        },
        ...props
    });
};

var Progress = /*#__PURE__*/Object.freeze({
  __proto__: null,
  Indicator: Indicator,
  Root: Root$1
});

const getRootStyles = (theme, useNewBorderColors)=>{
    return /*#__PURE__*/ css({
        alignItems: 'center',
        backgroundColor: theme.colors.backgroundSecondary,
        border: `1px solid ${useNewBorderColors ? theme.colors.border : theme.colors.borderDecorative}`,
        borderRadius: theme.borders.borderRadiusSm,
        boxShadow: theme.shadows.lg,
        display: 'flex',
        gap: theme.spacing.md,
        width: 'max-content',
        padding: theme.spacing.sm
    });
};
const Root = /*#__PURE__*/ forwardRef((props, ref)=>{
    const { theme } = useDesignSystemTheme();
    const useNewBorderColors = safex('databricks.fe.designsystem.useNewBorderColors', false);
    return /*#__PURE__*/ jsx(RadixToolbar.Root, {
        ...addDebugOutlineIfEnabled(),
        css: getRootStyles(theme, useNewBorderColors),
        ...props,
        ref: ref
    });
});
const Button = /*#__PURE__*/ forwardRef((props, ref)=>{
    return /*#__PURE__*/ jsx(RadixToolbar.Button, {
        ...props,
        ref: ref
    });
});
const getSeparatorStyles = (theme)=>{
    return /*#__PURE__*/ css({
        alignSelf: 'stretch',
        backgroundColor: theme.colors.borderDecorative,
        width: 1
    });
};
const Separator = /*#__PURE__*/ forwardRef((props, ref)=>{
    const { theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsx(RadixToolbar.Separator, {
        css: getSeparatorStyles(theme),
        ...props,
        ref: ref
    });
});
const Link = /*#__PURE__*/ forwardRef((props, ref)=>{
    return /*#__PURE__*/ jsx(RadixToolbar.Link, {
        ...props,
        ref: ref
    });
});
const ToggleGroup = /*#__PURE__*/ forwardRef((props, ref)=>{
    return /*#__PURE__*/ jsx(RadixToolbar.ToggleGroup, {
        ...props,
        ref: ref
    });
});
const getToggleItemStyles = (theme)=>{
    return /*#__PURE__*/ css({
        background: 'none',
        color: theme.colors.textPrimary,
        border: 'none',
        cursor: 'pointer',
        '&:hover': {
            color: theme.colors.actionDefaultTextHover
        },
        '&[data-state="on"]': {
            color: theme.colors.actionDefaultTextPress
        }
    });
};
const ToggleItem = /*#__PURE__*/ forwardRef((props, ref)=>{
    const { theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsx(RadixToolbar.ToggleItem, {
        css: getToggleItemStyles(theme),
        ...props,
        ref: ref
    });
});

var Toolbar = /*#__PURE__*/Object.freeze({
  __proto__: null,
  Button: Button,
  Link: Link,
  Root: Root,
  Separator: Separator,
  ToggleGroup: ToggleGroup,
  ToggleItem: ToggleItem
});

function treeGridReducer(state, action) {
    switch(action.type){
        case 'TOGGLE_ROW_EXPANDED':
            return {
                ...state,
                expandedRows: {
                    ...state.expandedRows,
                    [action.rowId]: !state.expandedRows[action.rowId]
                }
            };
        case 'SET_ACTIVE_ROW_ID':
            return {
                ...state,
                activeRowId: action.rowId
            };
        default:
            return state;
    }
}
function useDefaultTreeGridState({ initialState = {
    expandedRows: {}
} }) {
    const [state, dispatch] = useReducer(treeGridReducer, {
        ...initialState,
        activeRowId: null
    });
    const toggleRowExpanded = useCallback((rowId)=>{
        dispatch({
            type: 'TOGGLE_ROW_EXPANDED',
            rowId
        });
    }, []);
    const setActiveRowId = useCallback((rowId)=>{
        dispatch({
            type: 'SET_ACTIVE_ROW_ID',
            rowId
        });
    }, []);
    return {
        ...state,
        toggleRowExpanded,
        setActiveRowId
    };
}

const flattenData = (data, expandedRows, depth = 0, parentId = null)=>{
    return data.reduce((acc, node)=>{
        acc.push({
            ...node,
            depth,
            parentId
        });
        if (node.children && expandedRows[node.id]) {
            acc.push(...flattenData(node.children, expandedRows, depth + 1, node.id));
        }
        return acc;
    }, []);
};
const findFocusableElementForCellIndex = (row, cellIndex)=>{
    const cell = row.cells[cellIndex];
    return cell?.querySelector('[tabindex], button, a, input, select, textarea') || null;
};
const findNextFocusableCellIndexInRow = (row, columns, startIndex, direction)=>{
    const cells = Array.from(row.cells);
    const increment = direction === 'next' ? 1 : -1;
    const limit = direction === 'next' ? cells.length : -1;
    for(let i = startIndex + increment; i !== limit; i += increment){
        const cell = cells[i];
        const column = columns[i];
        const focusableElement = findFocusableElementForCellIndex(row, i);
        const cellContent = cell?.textContent?.trim();
        if (focusableElement || !column.contentFocusable && cellContent) {
            return i;
        }
    }
    return -1;
};
const TreeGrid = ({ data, columns, renderCell, renderRow, renderTable, renderHeader, onRowKeyboardSelect, onCellKeyboardSelect, includeHeader = false, state: providedState })=>{
    const defaultState = useDefaultTreeGridState({
        initialState: providedState && 'initialState' in providedState ? providedState.initialState : undefined
    });
    const { expandedRows, activeRowId, toggleRowExpanded, setActiveRowId } = providedState && !('initialState' in providedState) ? providedState : defaultState;
    const gridRef = useRef(null);
    const flattenedData = useMemo(()=>flattenData(data, expandedRows), [
        data,
        expandedRows
    ]);
    const focusRow = useCallback(({ rowId, rowIndex })=>{
        const row = gridRef.current?.querySelector(`tbody tr:nth-child(${rowIndex + 1})`);
        row?.focus();
        setActiveRowId(rowId);
    }, [
        setActiveRowId
    ]);
    const focusElement = useCallback((element, rowIndex)=>{
        if (element) {
            element.focus();
            setActiveRowId(flattenedData[rowIndex].id);
        }
    }, [
        setActiveRowId,
        flattenedData
    ]);
    const handleKeyDown = useCallback((event, rowIndex)=>{
        const { key } = event;
        let newRowIndex = rowIndex;
        const closestTd = event.target.closest('td');
        if (!gridRef.current || !gridRef.current.contains(document.activeElement)) {
            return;
        }
        const handleArrowVerticalNavigation = (direction)=>{
            if (closestTd) {
                const currentCellIndex = closestTd.cellIndex;
                let targetRow = closestTd.closest('tr')?.[`${direction}ElementSibling`];
                const moveFocusToRow = (row)=>{
                    const focusableElement = findFocusableElementForCellIndex(row, currentCellIndex);
                    const cellContent = row.cells[currentCellIndex]?.textContent?.trim();
                    if (focusableElement || !columns[currentCellIndex].contentFocusable && cellContent) {
                        event.preventDefault();
                        focusElement(focusableElement || row.cells[currentCellIndex], flattenedData.findIndex((r)=>r.id === row.dataset['id']));
                        return true;
                    }
                    return false;
                };
                while(targetRow){
                    if (moveFocusToRow(targetRow)) return;
                    targetRow = targetRow[`${direction}ElementSibling`];
                }
            } else if (document.activeElement instanceof HTMLTableRowElement) {
                if (direction === 'next') {
                    newRowIndex = Math.min(rowIndex + 1, flattenedData.length - 1);
                } else {
                    newRowIndex = Math.max(rowIndex - 1, 0);
                }
            }
        };
        const handleArrowHorizontalNavigation = (direction)=>{
            if (closestTd) {
                const currentRow = closestTd.closest('tr');
                let targetCellIndex = closestTd.cellIndex;
                targetCellIndex = findNextFocusableCellIndexInRow(currentRow, columns, targetCellIndex, direction);
                if (targetCellIndex !== -1) {
                    event.preventDefault();
                    const targetCell = currentRow.cells[targetCellIndex];
                    const focusableElement = findFocusableElementForCellIndex(currentRow, targetCellIndex);
                    focusElement(focusableElement || targetCell, rowIndex);
                    return;
                } else if (direction === 'previous' && targetCellIndex === -1) {
                    // If we're at the leftmost cell, focus on the row
                    event.preventDefault();
                    currentRow.focus();
                    return;
                }
            }
            if (document.activeElement instanceof HTMLTableRowElement) {
                const currentRow = document.activeElement;
                if (direction === 'next') {
                    if (flattenedData[rowIndex].children) {
                        if (!expandedRows[flattenedData[rowIndex].id]) {
                            toggleRowExpanded(flattenedData[rowIndex].id);
                        } else {
                            const firstCell = currentRow.cells[0];
                            focusElement(firstCell, rowIndex);
                        }
                    } else {
                        const firstFocusableCell = findNextFocusableCellIndexInRow(currentRow, columns, -1, 'next');
                        if (firstFocusableCell !== -1) {
                            focusElement(currentRow.cells[firstFocusableCell], rowIndex);
                        }
                    }
                } else {
                    if (expandedRows[flattenedData[rowIndex].id]) {
                        toggleRowExpanded(flattenedData[rowIndex].id);
                    } else if (flattenedData[rowIndex].depth && flattenedData[rowIndex].depth > 0) {
                        newRowIndex = flattenedData.findIndex((row)=>row.id === flattenedData[rowIndex].parentId);
                    }
                }
                return;
            }
            // If we're at the edge of the row, handle expanding/collapsing or moving to parent/child
            if (direction === 'next') {
                if (flattenedData[rowIndex].children && !expandedRows[flattenedData[rowIndex].id]) {
                    toggleRowExpanded(flattenedData[rowIndex].id);
                }
            } else {
                if (expandedRows[flattenedData[rowIndex].id]) {
                    toggleRowExpanded(flattenedData[rowIndex].id);
                } else if (flattenedData[rowIndex].depth && flattenedData[rowIndex].depth > 0) {
                    newRowIndex = flattenedData.findIndex((row)=>row.id === flattenedData[rowIndex].parentId);
                }
            }
        };
        const handleEnterKey = ()=>{
            if (closestTd) {
                onCellKeyboardSelect?.(flattenedData[rowIndex].id, columns[closestTd.cellIndex].id);
            } else if (document.activeElement instanceof HTMLTableRowElement) {
                onRowKeyboardSelect?.(flattenedData[rowIndex].id);
            }
        };
        switch(key){
            case 'ArrowUp':
                handleArrowVerticalNavigation('previous');
                break;
            case 'ArrowDown':
                handleArrowVerticalNavigation('next');
                break;
            case 'ArrowLeft':
                handleArrowHorizontalNavigation('previous');
                break;
            case 'ArrowRight':
                handleArrowHorizontalNavigation('next');
                break;
            case 'Enter':
                handleEnterKey();
                break;
            default:
                return;
        }
        if (newRowIndex !== rowIndex) {
            event.preventDefault();
            focusRow({
                rowId: flattenedData[newRowIndex].id,
                rowIndex: newRowIndex
            });
        }
    }, [
        expandedRows,
        columns,
        flattenedData,
        toggleRowExpanded,
        onRowKeyboardSelect,
        onCellKeyboardSelect,
        focusRow,
        focusElement
    ]);
    const defaultRenderRow = useCallback(({ rowProps, children })=>/*#__PURE__*/ jsx("tr", {
            ...rowProps,
            children: children
        }), []);
    const defaultRenderTable = useCallback(({ tableProps, children })=>/*#__PURE__*/ jsx("table", {
            ...tableProps,
            children: children
        }), []);
    const defaultRenderHeader = useCallback(({ columns, headerProps })=>/*#__PURE__*/ jsx("thead", {
            ...headerProps,
            children: /*#__PURE__*/ jsx("tr", {
                children: columns.map((column)=>/*#__PURE__*/ jsx("th", {
                        role: "columnheader",
                        children: column.header
                    }, column.id))
            })
        }), []);
    const renderRowWrapper = useCallback((row, rowIndex)=>{
        const isExpanded = expandedRows[row.id];
        const isKeyboardActive = row.id === (activeRowId ?? flattenedData[0].id);
        const rowProps = {
            key: row.id,
            'data-id': row.id,
            role: 'row',
            'aria-selected': false,
            'aria-level': (row.depth || 0) + 1,
            'aria-expanded': row.children ? isExpanded ? 'true' : 'false' : undefined,
            tabIndex: isKeyboardActive ? 0 : -1,
            onKeyDown: (e)=>handleKeyDown(e, rowIndex)
        };
        const children = columns.map((column, colIndex)=>{
            const cellProps = {
                key: `${row.id}-${column.id}`,
                role: column.isRowHeader ? 'rowheader' : 'gridcell',
                tabIndex: column.contentFocusable ? undefined : isKeyboardActive ? 0 : -1
            };
            return renderCell({
                row,
                column,
                rowDepth: row.depth || 0,
                rowIndex,
                colIndex,
                rowIsKeyboardActive: isKeyboardActive,
                rowIsExpanded: isExpanded,
                toggleRowExpanded,
                cellProps
            });
        });
        return (renderRow || defaultRenderRow)({
            row,
            rowIndex,
            isExpanded,
            isKeyboardActive,
            rowProps,
            children
        });
    }, [
        activeRowId,
        flattenedData,
        expandedRows,
        handleKeyDown,
        renderCell,
        renderRow,
        defaultRenderRow,
        toggleRowExpanded,
        columns
    ]);
    return (renderTable || defaultRenderTable)({
        tableProps: {
            role: 'treegrid',
            ref: gridRef
        },
        children: /*#__PURE__*/ jsxs(Fragment, {
            children: [
                includeHeader && (renderHeader || defaultRenderHeader)({
                    columns,
                    headerProps: {}
                }),
                /*#__PURE__*/ jsx("tbody", {
                    children: flattenedData.map((row, index)=>renderRowWrapper(row, index))
                })
            ]
        })
    });
};

export { BANNER_MAX_HEIGHT, BANNER_MIN_HEIGHT, Banner, DatePicker, Listbox, PillControl, Progress, RangePicker, Toolbar, TreeGrid, getDatePickerQuickActionBasic, getRangeQuickActionsBasic, useDefaultTreeGridState };
//# sourceMappingURL=development.js.map
