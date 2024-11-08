import React__default, { useMemo, forwardRef, useState, useRef, useEffect, useCallback, useImperativeHandle, createContext, useReducer } from 'react';
import { jsx, Fragment, jsxs } from '@emotion/react/jsx-runtime';
import { css } from '@emotion/react';
import { W as WarningIcon, y as DangerIcon, b as DesignSystemEventProviderAnalyticsEventTypes, a as useDesignSystemTheme, c as useDesignSystemEventComponentCallbacks, d as DesignSystemEventProviderComponentTypes, f as useNotifyOnFirstView, B as Button$1, C as CloseIcon, h as addDebugOutlineIfEnabled, p as Typography, a6 as primitiveColors, k as Root$4, T as Trigger, l as Content, G as ChevronLeftIcon, m as ChevronRightIcon, q as importantify } from './Typography-C4ciIwWZ.js';
export { a8 as Form, a7 as useFormContext } from './Typography-C4ciIwWZ.js';
import { M as MegaphoneIcon, I as Input, C as ClockIcon } from './index-D9gS2nVh.js';
import { startOfToday, startOfYesterday, sub, isValid, format, startOfWeek, endOfToday, endOfYesterday, isAfter, isBefore } from 'date-fns';
import { DayPicker, useDayRender, Button as Button$2 } from 'react-day-picker';
import { RadioGroup, RadioGroupItem } from '@radix-ui/react-radio-group';
import * as Progress$1 from '@radix-ui/react-progress';
import * as RadixSlider from '@radix-ui/react-slider';
export { S as Stepper } from './Stepper-D_5J10NP.js';
import * as RadixToolbar from '@radix-ui/react-toolbar';
import 'antd';
import '@radix-ui/react-popover';
import '@radix-ui/react-tooltip';
import '@ant-design/icons';
import 'lodash/isNil';
import 'lodash/endsWith';
import 'lodash/isBoolean';
import 'lodash/isNumber';
import 'lodash/isString';
import 'lodash/mapValues';
import 'lodash/memoize';
import '@emotion/unitless';
import 'lodash/isUndefined';

function _EMOTION_STRINGIFIED_CSS_ERROR__$2() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const {
  Text,
  Paragraph
} = Typography;
const BANNER_MIN_HEIGHT = 68;
// Max height will allow 2 lines of description (3 lines total)
const BANNER_MAX_HEIGHT = 82;
var _ref$1 = process.env.NODE_ENV === "production" ? {
  name: "1te1whx",
  styles: "margin-left:auto;display:flex;align-items:center"
} : {
  name: "13c4h59-rightContainer",
  styles: "margin-left:auto;display:flex;align-items:center;label:rightContainer;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$2
};
const useStyles = (props, theme) => {
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
    banner: /*#__PURE__*/css("max-height:", BANNER_MAX_HEIGHT, "px;display:flex;align-items:center;width:100%;padding:8px;box-sizing:border-box;background-color:", colorScheme.backgroundDefaultColor, ";border:1px solid ", colorScheme.borderDefaultColor, ";" + (process.env.NODE_ENV === "production" ? "" : ";label:banner;")),
    iconContainer: /*#__PURE__*/css("display:flex;color:", colorScheme.iconColor ? colorScheme.iconColor : colorScheme.textColor, ";align-self:", props.description ? 'flex-start' : 'center', ";box-sizing:border-box;max-width:60px;padding-top:4px;padding-bottom:4px;padding-right:", theme.spacing.xs, "px;" + (process.env.NODE_ENV === "production" ? "" : ";label:iconContainer;")),
    mainContent: /*#__PURE__*/css("flex-direction:column;align-self:", props.description ? 'flex-start' : 'center', ";display:flex;box-sizing:border-box;padding-right:", theme.spacing.sm, "px;padding-top:2px;padding-bottom:2px;min-width:", theme.spacing.lg, "px;width:100%;" + (process.env.NODE_ENV === "production" ? "" : ";label:mainContent;")),
    messageTextBlock: /*#__PURE__*/css("display:-webkit-box;-webkit-line-clamp:1;-webkit-box-orient:vertical;overflow:hidden;&&{color:", colorScheme.textColor, ";}" + (process.env.NODE_ENV === "production" ? "" : ";label:messageTextBlock;")),
    descriptionBlock: /*#__PURE__*/css("display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;&&{color:", colorScheme.textColor, ";}" + (process.env.NODE_ENV === "production" ? "" : ";label:descriptionBlock;")),
    rightContainer: _ref$1,
    closeIconContainer: /*#__PURE__*/css("display:flex;margin-left:", theme.spacing.xs, "px;box-sizing:border-box;line-height:0;" + (process.env.NODE_ENV === "production" ? "" : ";label:closeIconContainer;")),
    closeButton: /*#__PURE__*/css("cursor:pointer;background:none;border:none;margin:0;&&{height:24px!important;width:24px!important;padding:", theme.spacing.xs, "px!important;}&&:hover{background-color:transparent!important;border-color:", colorScheme.textHoverColor, "!important;color:", colorScheme.closeIconTextHoverColor ? colorScheme.closeIconTextHoverColor : colorScheme.textColor, "!important;background-color:", colorScheme.closeIconBackgroundHoverColor ? colorScheme.closeIconBackgroundHoverColor : colorScheme.backgroundDefaultColor, "!important;}&&:active{border-color:", colorScheme.actionBorderColor, "!important;color:", colorScheme.closeIconTextPressColor ? colorScheme.closeIconTextPressColor : colorScheme.textColor, "!important;background-color:", colorScheme.closeIconBackgroundPressColor ? colorScheme.closeIconBackgroundPressColor : colorScheme.backgroundDefaultColor, "!important;}" + (process.env.NODE_ENV === "production" ? "" : ";label:closeButton;")),
    closeIcon: /*#__PURE__*/css("color:", colorScheme.closeIconColor ? colorScheme.closeIconColor : colorScheme.textColor, "!important;" + (process.env.NODE_ENV === "production" ? "" : ";label:closeIcon;")),
    actionButtonContainer: /*#__PURE__*/css("margin-right:", theme.spacing.xs, "px;" + (process.env.NODE_ENV === "production" ? "" : ";label:actionButtonContainer;")),
    // Override design system colors to show the use the action text color for text and border.
    // Also overrides text for links.
    actionButton: /*#__PURE__*/css("color:", colorScheme.textColor, "!important;border-color:", colorScheme.actionBorderColor ? colorScheme.actionBorderColor : colorScheme.textColor, "!important;&:focus,&:hover{border-color:", colorScheme.actionButtonBorderHoverColor ? colorScheme.actionButtonBorderHoverColor : colorScheme.textHoverColor, "!important;color:", colorScheme.textColor, "!important;background-color:", colorScheme.actionButtonBackgroundHoverColor, "!important;}&:active{border-color:", colorScheme.actionButtonBorderPressColor ? colorScheme.actionButtonBorderPressColor : colorScheme.actionBorderColor, "!important;color:", colorScheme.textPressColor, "!important;background-color:", colorScheme.actionButtonBackgroundPressColor, "!important;}a{color:", theme.colors.actionPrimaryTextDefault, ";}a:focus,a:hover{color:", colorScheme.textHoverColor, ";text-decoration:none;}a:active{color:", colorScheme.textPressColor, " text-decoration:none;}" + (process.env.NODE_ENV === "production" ? "" : ";label:actionButton;"))
  };
};
const levelToIconMap = {
  info: jsx(MegaphoneIcon, {
    "data-testid": "level-info-icon"
  }),
  info_light_purple: jsx(MegaphoneIcon, {
    "data-testid": "level-info-light-purple-icon"
  }),
  info_dark_purple: jsx(MegaphoneIcon, {
    "data-testid": "level-info-dark-purple-icon"
  }),
  warning: jsx(WarningIcon, {
    "data-testid": "level-warning-icon"
  }),
  error: jsx(DangerIcon, {
    "data-testid": "level-error-icon"
  })
};
const Banner = props => {
  const {
    level,
    message,
    description,
    ctaText,
    onAccept,
    closable,
    onClose,
    closeButtonAriaLabel,
    componentId,
    analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnView]
  } = props;
  const [closed, setClosed] = React__default.useState(false);
  const {
    theme
  } = useDesignSystemTheme();
  const styles = useStyles(props, theme);
  const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.Banner,
    componentId,
    analyticsEvents: memoizedAnalyticsEvents
  });
  const {
    elementRef
  } = useNotifyOnFirstView({
    onView: eventContext.onView
  });
  const actionButton = onAccept && ctaText ? jsx("div", {
    css: styles.actionButtonContainer,
    children: jsx(Button$1, {
      componentId: `${componentId}.accept`,
      onClick: onAccept,
      css: styles.actionButton,
      size: "small",
      children: ctaText
    })
  }) : null;
  const close = closable !== false ? jsx("div", {
    css: styles.closeIconContainer,
    children: jsx(Button$1, {
      componentId: `${componentId}.close`,
      css: styles.closeButton,
      onClick: () => {
        if (onClose) {
          onClose();
        }
        setClosed(true);
      },
      "aria-label": closeButtonAriaLabel !== null && closeButtonAriaLabel !== void 0 ? closeButtonAriaLabel : 'Close',
      "data-testid": "banner-dismiss",
      children: jsx(CloseIcon, {
        css: styles.closeIcon
      })
    })
  }) : null;
  return jsx(Fragment, {
    children: !closed && jsxs("div", {
      ref: elementRef,
      ...addDebugOutlineIfEnabled(),
      css: styles.banner,
      className: "banner",
      "data-testid": props['data-testid'],
      role: "alert",
      children: [jsx("div", {
        css: styles.iconContainer,
        children: levelToIconMap[level]
      }), jsxs("div", {
        css: styles.mainContent,
        children: [jsx(Text, {
          size: "md",
          bold: true,
          css: styles.messageTextBlock,
          title: message,
          children: message
        }), description && jsx(Paragraph, {
          withoutMargins: true,
          css: styles.descriptionBlock,
          title: description,
          children: description
        })]
      }), jsxs("div", {
        css: styles.rightContainer,
        children: [actionButton, close]
      })]
    })
  });
};

const getDayPickerStyles = (prefix, theme) => /*#__PURE__*/css(".", prefix, "{--rdp-cell-size:", theme.general.heightSm, "px;--rdp-caption-font-size:", theme.typography.fontSizeBase, "px;--rdp-accent-color:", theme.colors.actionPrimaryBackgroundDefault, ";--rdp-background-color:", theme.colors.actionTertiaryBackgroundPress, ";--rdp-accent-color-dark:", theme.colors.actionPrimaryBackgroundDefault, ";--rdp-background-color-dark:", theme.colors.actionTertiaryBackgroundPress, ";--rdp-outline:2px solid var(--rdp-accent-color);--rdp-outline-selected:3px solid var(--rdp-accent-color);--rdp-selected-color:#fff;padding:4px;}.", prefix, "-vhidden{box-sizing:border-box;padding:0;margin:0;background:transparent;border:0;-moz-appearance:none;-webkit-appearance:none;appearance:none;position:absolute!important;top:0;width:1px!important;height:1px!important;padding:0!important;overflow:hidden!important;clip:rect(1px, 1px, 1px, 1px)!important;border:0!important;}.", prefix, "-button_reset{appearance:none;position:relative;margin:0;padding:0;cursor:default;color:inherit;background:none;font:inherit;-moz-appearance:none;-webkit-appearance:none;}.", prefix, "-button_reset:focus-visible{outline:none;}.", prefix, "-button{border:2px solid transparent;}.", prefix, "-button[disabled]:not(.", prefix, "-day_selected){opacity:0.25;}.", prefix, "-button:not([disabled]){cursor:pointer;}.", prefix, "-button:focus-visible:not([disabled]){color:inherit;background-color:var(--rdp-background-color);border:var(--rdp-outline);}.", prefix, "-button:hover:not([disabled]):not(.", prefix, "-day_selected){background-color:var(--rdp-background-color);}.", prefix, "-months{display:flex;justify-content:center;}.", prefix, "-month{margin:0 1em;}.", prefix, "-month:first-of-type{margin-left:0;}.", prefix, "-month:last-child{margin-right:0;}.", prefix, "-table{margin:0;max-width:calc(var(--rdp-cell-size) * 7);border-collapse:collapse;}.", prefix, "-with_weeknumber .", prefix, "-table{max-width:calc(var(--rdp-cell-size) * 8);border-collapse:collapse;}.", prefix, "-caption{display:flex;align-items:center;justify-content:space-between;padding:0;text-align:left;}.", prefix, "-multiple_months .", prefix, "-caption{position:relative;display:block;text-align:center;}.", prefix, "-caption_dropdowns{position:relative;display:inline-flex;}.", prefix, "-caption_label{position:relative;z-index:1;display:inline-flex;align-items:center;margin:0;padding:0 0.25em;white-space:nowrap;color:currentColor;border:0;border:2px solid transparent;font-family:inherit;font-size:var(--rdp-caption-font-size);font-weight:600;}.", prefix, "-nav{white-space:nowrap;}.", prefix, "-multiple_months .", prefix, "-caption_start .", prefix, "-nav{position:absolute;top:50%;left:0;transform:translateY(-50%);}.", prefix, "-multiple_months .", prefix, "-caption_end .", prefix, "-nav{position:absolute;top:50%;right:0;transform:translateY(-50%);}.", prefix, "-nav_button{display:inline-flex;align-items:center;justify-content:center;width:var(--rdp-cell-size);height:var(--rdp-cell-size);}.", prefix, "-dropdown_year,.", prefix, "-dropdown_month{position:relative;display:inline-flex;align-items:center;}.", prefix, "-dropdown{appearance:none;position:absolute;z-index:2;top:0;bottom:0;left:0;width:100%;margin:0;padding:0;cursor:inherit;opacity:0;border:none;background-color:transparent;font-family:inherit;font-size:inherit;line-height:inherit;}.", prefix, "-dropdown[disabled]{opacity:unset;color:unset;}.", prefix, "-dropdown:focus-visible:not([disabled])+.", prefix, "-caption_label{background-color:var(--rdp-background-color);border:var(--rdp-outline);border-radius:6px;}.", prefix, "-dropdown_icon{margin:0 0 0 5px;}.", prefix, "-head{border:0;}.", prefix, "-head_row,.", prefix, "-row{height:100%;}.", prefix, "-head_cell{vertical-align:middle;font-size:inherit;font-weight:400;color:", theme.colors.textSecondary, ";text-align:center;height:100%;height:var(--rdp-cell-size);padding:0;text-transform:uppercase;}.", prefix, "-tbody{border:0;}.", prefix, "-tfoot{margin:0.5em;}.", prefix, "-cell{width:var(--rdp-cell-size);height:100%;height:var(--rdp-cell-size);padding:0;text-align:center;}.", prefix, "-weeknumber{font-size:0.75em;}.", prefix, "-weeknumber,.", prefix, "-day{display:flex;overflow:hidden;align-items:center;justify-content:center;box-sizing:border-box;width:var(--rdp-cell-size);max-width:var(--rdp-cell-size);height:var(--rdp-cell-size);margin:0;border:2px solid transparent;border-radius:", theme.general.borderRadiusBase, "px;}.", prefix, "-day_today:not(.", prefix, "-day_outside){font-weight:bold;}.", prefix, "-day_selected,.", prefix, "-day_selected:focus-visible,.", prefix, "-day_selected:hover{color:var(--rdp-selected-color);opacity:1;background-color:var(--rdp-accent-color);}.", prefix, "-day_outside{pointer-events:none;color:", theme.colors.textSecondary, ";}.", prefix, "-day_selected:focus-visible{outline:var(--rdp-outline);outline-offset:2px;z-index:1;}.", prefix, ":not([dir='rtl']) .", prefix, "-day_range_start:not(.", prefix, "-day_range_end){border-top-right-radius:0;border-bottom-right-radius:0;}.", prefix, ":not([dir='rtl']) .", prefix, "-day_range_end:not(.", prefix, "-day_range_start){border-top-left-radius:0;border-bottom-left-radius:0;}.", prefix, "[dir='rtl'] .", prefix, "-day_range_start:not(.", prefix, "-day_range_end){border-top-left-radius:0;border-bottom-left-radius:0;}.", prefix, "[dir='rtl'] .", prefix, "-day_range_end:not(.", prefix, "-day_range_start){border-top-right-radius:0;border-bottom-right-radius:0;}.", prefix, "-day_range_start,.", prefix, "-day_range_end{border:0;&>span{width:100%;height:100%;display:flex;align-items:center;justify-content:center;border-radius:", theme.general.borderRadiusBase, "px;background-color:var(--rdp-accent-color);color:", theme.colors.white, ";}}.", prefix, "-day_range_end.", prefix, "-day_range_start{border-radius:", theme.general.borderRadiusBase, "px;}.", prefix, "-day_range_middle{border-radius:0;background-color:var(--rdp-background-color);color:", theme.colors.actionDefaultTextDefault, ";&:hover{color:", theme.colors.actionTertiaryTextHover, ";}}.", prefix, "-row>td:last-of-type .", prefix, "-day_range_middle{border-top-right-radius:", theme.general.borderRadiusBase, "px;border-bottom-right-radius:", theme.general.borderRadiusBase, "px;}.", prefix, "-row>td:first-of-type .", prefix, "-day_range_middle{border-top-left-radius:", theme.general.borderRadiusBase, "px;border-bottom-left-radius:", theme.general.borderRadiusBase, "px;}" + (process.env.NODE_ENV === "production" ? "" : ";label:getDayPickerStyles;"));

const generateDatePickerClassNames = prefix => ({
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

function _EMOTION_STRINGIFIED_CSS_ERROR__$1() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
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
    return jsx("div", {
      role: "cell"
    });
  }
  if (!dayRender.isButton) {
    return jsx("div", {
      ...dayRender.divProps
    });
  }
  return jsx(Button$2, {
    name: "day",
    ref: buttonRef,
    ...dayRender.buttonProps,
    role: "button"
  });
}
const getDatePickerQuickActionBasic = _ref => {
  let {
    today,
    yesterday,
    sevenDaysAgo
  } = _ref;
  return [{
    label: 'Today',
    value: startOfToday(),
    ...today
  }, {
    label: 'Yesterday',
    value: startOfYesterday(),
    ...yesterday
  }, {
    label: '7 days ago',
    value: sub(startOfToday(), {
      days: 7
    }),
    ...sevenDaysAgo
  }];
};
const DatePicker = /*#__PURE__*/forwardRef((props, ref) => {
  const {
    classNamePrefix,
    theme
  } = useDesignSystemTheme();
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
    ...restProps
  } = props;
  const format$1 = includeTime ? 'yyyy-MM-dd HH:mm' : 'yyyy-MM-dd';
  const [date, setDate] = useState(value);
  const [isVisible, setIsVisible] = useState(Boolean(open));
  const inputRef = useRef(null);
  const visibleRef = useRef(isVisible);
  // Needed to avoid the clear icon click also reopening the datepicker
  const fromClearRef = useRef(null);
  useEffect(() => {
    if (!isVisible && visibleRef.current) {
      var _inputRef$current;
      (_inputRef$current = inputRef.current) === null || _inputRef$current === void 0 || _inputRef$current.focus();
    }
    visibleRef.current = isVisible;
    onOpenChange === null || onOpenChange === void 0 || onOpenChange(isVisible);
  }, [isVisible, onOpenChange]);
  useEffect(() => {
    setIsVisible(Boolean(open));
  }, [open]);
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
  const handleChange = useCallback((date, isCalendarUpdate) => {
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
  }, [onChange, name]);
  const handleDatePickerUpdate = date => {
    setDate(prevDate => {
      // Set default time if date is set the first time
      if (!prevDate && date && includeTime && defaultTime) {
        const timeSplit = defaultTime.split(':');
        date === null || date === void 0 || date.setHours(+timeSplit[0]);
        date === null || date === void 0 || date.setMinutes(+timeSplit[1]);
      } else if (prevDate && date && includeTime) {
        date.setHours(prevDate.getHours());
        date.setMinutes(prevDate.getMinutes());
      }
      handleChange === null || handleChange === void 0 || handleChange(date, true);
      return date;
    });
    if (!includeTime) {
      setIsVisible(false);
    }
  };
  const handleInputUpdate = updatedDate => {
    if (!updatedDate || !isValid(updatedDate)) {
      setDate(undefined);
      handleChange === null || handleChange === void 0 || handleChange(undefined, false);
      return;
    }
    if (date && updatedDate && includeTime) {
      updatedDate.setHours(updatedDate.getHours());
      updatedDate.setMinutes(updatedDate.getMinutes());
    }
    setDate(updatedDate);
    handleChange === null || handleChange === void 0 || handleChange(updatedDate, false);
    if (!includeTime) {
      setIsVisible(false);
    }
  };
  const handleClear = useCallback(() => {
    setDate(undefined);
    onClear === null || onClear === void 0 || onClear();
    handleChange === null || handleChange === void 0 || handleChange(undefined, false);
  }, [onClear, handleChange]);
  const handleTimeUpdate = e => {
    var _e$nativeEvent;
    const newTime = (_e$nativeEvent = e.nativeEvent) === null || _e$nativeEvent === void 0 || (_e$nativeEvent = _e$nativeEvent.target) === null || _e$nativeEvent === void 0 ? void 0 : _e$nativeEvent.value;
    const time = date && isValid(date) ? format(date, 'HH:mm') : undefined;
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
      var _input;
      const clearIcon = (_input = inputRef.current.input) === null || _input === void 0 || (_input = _input.closest('[type="button"]')) === null || _input === void 0 ? void 0 : _input.querySelector(`.${classNamePrefix}-input-clear-icon`);
      if (clearIcon !== fromClearRef.current) {
        fromClearRef.current = clearIcon;
        const clientEventListener = e => {
          e.stopPropagation();
          e.preventDefault();
          handleClear();
        };
        clearIcon.addEventListener('click', clientEventListener);
      }
    }
  }, [classNamePrefix, defaultTime, handleClear, allowClear]);
  const {
    classNames,
    datePickerStyles
  } = useMemo(() => ({
    classNames: generateDatePickerClassNames(`${classNamePrefix}-datepicker`),
    datePickerStyles: getDayPickerStyles(`${classNamePrefix}-datepicker`, theme)
  }), [classNamePrefix, theme]);
  const chevronLeftIconComp = props => jsx(ChevronLeftIcon, {
    ...props
  });
  const chevronRightIconComp = props => jsx(ChevronRightIcon, {
    ...props
  });
  return jsx("div", {
    className: `${classNamePrefix}-datepicker`,
    css: /*#__PURE__*/css({
      width,
      minWidth,
      maxWidth,
      pointerEvents: restProps !== null && restProps !== void 0 && restProps.disabled ? 'none' : 'auto'
    }, process.env.NODE_ENV === "production" ? "" : ";label:DatePicker;"),
    ...wrapperProps,
    children: jsxs(Root$4, {
      componentId: "codegen_design-system_src_development_datepicker_datepicker.tsx_330",
      open: isVisible,
      onOpenChange: setIsVisible,
      children: [jsx(Trigger, {
        asChild: true,
        disabled: restProps === null || restProps === void 0 ? void 0 : restProps.disabled,
        role: "combobox",
        children: jsxs("div", {
          children: [jsx(Input, {
            id: id,
            ref: inputRef,
            name: name,
            validationState: validationState,
            allowClear: allowClear,
            placeholder: "Select Date",
            "aria-label": includeTime ? 'Select Date and Time' : 'Select Date',
            prefix: "Date:",
            role: "textbox",
            max: includeTime ? '9999-12-31T23:59' : '9999-12-31',
            ...restProps,
            css: /*#__PURE__*/css({
              '*::-webkit-calendar-picker-indicator': {
                display: 'none'
              },
              [`.${classNamePrefix}-input-prefix`]: {
                ...(!(restProps !== null && restProps !== void 0 && restProps.disabled) && {
                  color: `${theme.colors.textPrimary} !important`
                })
              },
              [`&.${classNamePrefix}-input-affix-wrapper > *`]: {
                height: theme.typography.lineHeightBase
              }
            }, process.env.NODE_ENV === "production" ? "" : ";label:DatePicker;"),
            type: includeTime ? 'datetime-local' : 'date',
            onKeyDown: event => handleInputKeyDown(event, setIsVisible),
            onChange: e => handleInputUpdate(new Date(e.target.value)),
            value: date && isValid(date) ? format(date, format$1) : undefined
          }), jsx("input", {
            type: "hidden",
            ref: ref,
            value: date || ''
          })]
        })
      }), jsxs(Content, {
        align: "start",
        css: datePickerStyles,
        children: [jsx(DayPicker, {
          initialFocus: true,
          ...datePickerProps,
          mode: mode,
          selected: mode === 'range' ? selected : date,
          onDayClick: handleDatePickerUpdate,
          showOutsideDays: mode === 'range' ? false : true,
          formatters: {
            formatWeekdayName: date => format(date, 'iiiii', {
              locale: datePickerProps === null || datePickerProps === void 0 ? void 0 : datePickerProps.locale
            })
          },
          components: {
            Day,
            IconLeft: chevronLeftIconComp,
            IconRight: chevronRightIconComp
          },
          defaultMonth: date,
          classNames: classNames
        }), (quickActions === null || quickActions === void 0 ? void 0 : quickActions.length) && jsx("div", {
          style: {
            display: 'flex',
            gap: theme.spacing.sm,
            marginBottom: theme.spacing.md,
            padding: `${theme.spacing.xs}px ${theme.spacing.xs}px 0`,
            maxWidth: 225,
            flexWrap: 'wrap'
          },
          children: quickActions === null || quickActions === void 0 ? void 0 : quickActions.map((action, i) => jsx(Button$1, {
            size: "small",
            componentId: "codegen_design-system_src_design-system_datepicker_datepicker.tsx_281",
            onClick: () => action.onClick ? action.onClick(action.value) : !Array.isArray(action.value) && handleDatePickerUpdate(action.value),
            children: action.label
          }, i))
        }), includeTime && jsx(Input, {
          componentId: "codegen_design-system_src_development_datepicker_datepicker.tsx_306",
          type: "time",
          "aria-label": "Time",
          role: "textbox",
          ...timeInputProps,
          value: date && isValid(date) ? format(date, 'HH:mm') : undefined,
          onChange: handleTimeUpdate,
          css: /*#__PURE__*/css({
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
          }, process.env.NODE_ENV === "production" ? "" : ";label:DatePicker;"),
          suffix: jsx(ClockIcon, {}),
          disabled: timeInputProps === null || timeInputProps === void 0 ? void 0 : timeInputProps.disabled
        }), mode === 'range' && includeTime && onOkPress && jsx("div", {
          css: /*#__PURE__*/css({
            paddingTop: theme.spacing.md,
            display: 'flex',
            justifyContent: 'flex-end'
          }, process.env.NODE_ENV === "production" ? "" : ";label:DatePicker;"),
          children: jsx(Button$1, {
            "aria-label": "Open end date picker",
            type: "primary",
            componentId: "datepicker-dubois-ok-button",
            onClick: onOkPress,
            children: okButtonLabel !== null && okButtonLabel !== void 0 ? okButtonLabel : 'Ok'
          })
        })]
      })]
    })
  });
});
const getRangeQuickActionsBasic = _ref2 => {
  let {
    today,
    yesterday,
    lastWeek
  } = _ref2;
  const todayStart = startOfToday();
  const weekStart = startOfWeek(todayStart);
  return [{
    label: 'Today',
    value: [todayStart, endOfToday()],
    ...today
  }, {
    label: 'Yesterday',
    value: [startOfYesterday(), endOfYesterday()],
    ...yesterday
  }, {
    label: 'Last week',
    value: [sub(weekStart, {
      days: 7
    }), sub(weekStart, {
      days: 1
    })],
    ...lastWeek
  }];
};
var _ref3 = process.env.NODE_ENV === "production" ? {
  name: "stj4fv",
  styles: "*::-webkit-calendar-picker-indicator{display:none;}border-top-right-radius:0;border-bottom-right-radius:0"
} : {
  name: "p55v8c-RangePicker",
  styles: "*::-webkit-calendar-picker-indicator{display:none;}border-top-right-radius:0;border-bottom-right-radius:0;label:RangePicker;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$1
};
const RangePicker = props => {
  var _startDatePickerProps6, _endDatePickerProps$d5, _range$from;
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
    wrapperProps
  } = props;
  const [range, setRange] = useState({
    from: startDatePickerProps === null || startDatePickerProps === void 0 ? void 0 : startDatePickerProps.value,
    to: endDatePickerProps === null || endDatePickerProps === void 0 ? void 0 : endDatePickerProps.value
  });
  const {
    classNamePrefix
  } = useDesignSystemTheme();

  // Focus is lost when the popover is closed, we need to set the focus back to the input that opened the popover manually.
  const [isFromVisible, setIsFromVisible] = useState(false);
  const [isToVisible, setIsToVisible] = useState(false);
  const [isRangeInputFocused, setIsRangeInputFocused] = useState(false);
  const fromInputRef = useRef(null);
  const toInputRef = useRef(null);
  useImperativeHandle(startDatePickerProps === null || startDatePickerProps === void 0 ? void 0 : startDatePickerProps.ref, () => fromInputRef.current);
  useImperativeHandle(endDatePickerProps === null || endDatePickerProps === void 0 ? void 0 : endDatePickerProps.ref, () => toInputRef.current);
  const fromInputRefVisible = useRef(isFromVisible);
  const toInputRefVisible = useRef(isToVisible);
  useEffect(() => {
    if (!isFromVisible && fromInputRefVisible.current) {
      var _fromInputRef$current;
      (_fromInputRef$current = fromInputRef.current) === null || _fromInputRef$current === void 0 || _fromInputRef$current.focus();
    }
    fromInputRefVisible.current = isFromVisible;
  }, [isFromVisible]);
  useEffect(() => {
    if (!isToVisible && toInputRefVisible.current) {
      var _toInputRef$current;
      (_toInputRef$current = toInputRef.current) === null || _toInputRef$current === void 0 || _toInputRef$current.focus();
    }
    toInputRefVisible.current = isToVisible;
  }, [isToVisible]);
  const checkIfDateTimeIsDisabled = useCallback(function (date) {
    let isStart = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : false;
    const dateToCompareTo = isStart ? range === null || range === void 0 ? void 0 : range.to : range === null || range === void 0 ? void 0 : range.from;
    if (date && dateToCompareTo) {
      return isStart ? isAfter(date, dateToCompareTo) : isBefore(date, dateToCompareTo);
    }
    return false;
  }, [range]);
  useEffect(() => {
    setRange(prevValue => ({
      from: startDatePickerProps === null || startDatePickerProps === void 0 ? void 0 : startDatePickerProps.value,
      to: prevValue === null || prevValue === void 0 ? void 0 : prevValue.to
    }));
  }, [startDatePickerProps === null || startDatePickerProps === void 0 ? void 0 : startDatePickerProps.value]);
  useEffect(() => {
    setRange(prevValue => ({
      from: prevValue === null || prevValue === void 0 ? void 0 : prevValue.from,
      to: endDatePickerProps === null || endDatePickerProps === void 0 ? void 0 : endDatePickerProps.value
    }));
  }, [endDatePickerProps === null || endDatePickerProps === void 0 ? void 0 : endDatePickerProps.value]);
  const quickActionsWithHandler = useMemo(() => {
    if (quickActions) {
      return quickActions.map(action => {
        if (Array.isArray(action.value)) {
          return {
            ...action,
            onClick: value => {
              var _action$onClick;
              setRange({
                from: value[0],
                to: value[1]
              });
              onChange === null || onChange === void 0 || onChange({
                from: value[0],
                to: value[1]
              });
              (_action$onClick = action.onClick) === null || _action$onClick === void 0 || _action$onClick.call(action, value);
              setIsFromVisible(false);
              setIsToVisible(false);
            }
          };
        }
        return action;
      });
    }
    return quickActions;
  }, [quickActions, onChange]);
  const handleUpdateDate = useCallback((e, isStart) => {
    const date = e.target.value;
    const newRange = isStart ? {
      from: date,
      to: range === null || range === void 0 ? void 0 : range.to
    } : {
      from: range === null || range === void 0 ? void 0 : range.from,
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
      var _startDatePickerProps;
      startDatePickerProps === null || startDatePickerProps === void 0 || (_startDatePickerProps = startDatePickerProps.onChange) === null || _startDatePickerProps === void 0 || _startDatePickerProps.call(startDatePickerProps, e);
    } else {
      var _endDatePickerProps$o;
      endDatePickerProps === null || endDatePickerProps === void 0 || (_endDatePickerProps$o = endDatePickerProps.onChange) === null || _endDatePickerProps$o === void 0 || _endDatePickerProps$o.call(endDatePickerProps, e);
    }
    setRange(newRange);
    onChange === null || onChange === void 0 || onChange(newRange);
  }, [onChange, includeTime, startDatePickerProps, endDatePickerProps, range]);

  // Use useMemo to calculate disabled dates
  const disabledDates = useMemo(() => {
    var _startDatePickerProps2, _endDatePickerProps$d;
    let startDisabledFromProps, endDisabledFromProps;
    if (startDatePickerProps !== null && startDatePickerProps !== void 0 && (_startDatePickerProps2 = startDatePickerProps.datePickerProps) !== null && _startDatePickerProps2 !== void 0 && _startDatePickerProps2.disabled) {
      var _startDatePickerProps3, _startDatePickerProps4, _startDatePickerProps5;
      startDisabledFromProps = Array.isArray(startDatePickerProps === null || startDatePickerProps === void 0 || (_startDatePickerProps3 = startDatePickerProps.datePickerProps) === null || _startDatePickerProps3 === void 0 ? void 0 : _startDatePickerProps3.disabled) ? startDatePickerProps === null || startDatePickerProps === void 0 || (_startDatePickerProps4 = startDatePickerProps.datePickerProps) === null || _startDatePickerProps4 === void 0 ? void 0 : _startDatePickerProps4.disabled : [startDatePickerProps === null || startDatePickerProps === void 0 || (_startDatePickerProps5 = startDatePickerProps.datePickerProps) === null || _startDatePickerProps5 === void 0 ? void 0 : _startDatePickerProps5.disabled];
    }
    const startDisabled = [{
      after: range === null || range === void 0 ? void 0 : range.to
    }, ...(startDisabledFromProps ? startDisabledFromProps : [])].filter(Boolean);
    if (endDatePickerProps !== null && endDatePickerProps !== void 0 && (_endDatePickerProps$d = endDatePickerProps.datePickerProps) !== null && _endDatePickerProps$d !== void 0 && _endDatePickerProps$d.disabled) {
      var _endDatePickerProps$d2, _endDatePickerProps$d3, _endDatePickerProps$d4;
      endDisabledFromProps = Array.isArray(endDatePickerProps === null || endDatePickerProps === void 0 || (_endDatePickerProps$d2 = endDatePickerProps.datePickerProps) === null || _endDatePickerProps$d2 === void 0 ? void 0 : _endDatePickerProps$d2.disabled) ? endDatePickerProps === null || endDatePickerProps === void 0 || (_endDatePickerProps$d3 = endDatePickerProps.datePickerProps) === null || _endDatePickerProps$d3 === void 0 ? void 0 : _endDatePickerProps$d3.disabled : [endDatePickerProps === null || endDatePickerProps === void 0 || (_endDatePickerProps$d4 = endDatePickerProps.datePickerProps) === null || _endDatePickerProps$d4 === void 0 ? void 0 : _endDatePickerProps$d4.disabled];
    }
    const endDisabled = [{
      before: range === null || range === void 0 ? void 0 : range.from
    }, ...(endDisabledFromProps ? endDisabledFromProps : [])].filter(Boolean);
    return {
      startDisabled,
      endDisabled
    };
  }, [range === null || range === void 0 ? void 0 : range.from, range === null || range === void 0 ? void 0 : range.to, startDatePickerProps === null || startDatePickerProps === void 0 || (_startDatePickerProps6 = startDatePickerProps.datePickerProps) === null || _startDatePickerProps6 === void 0 ? void 0 : _startDatePickerProps6.disabled, endDatePickerProps === null || endDatePickerProps === void 0 || (_endDatePickerProps$d5 = endDatePickerProps.datePickerProps) === null || _endDatePickerProps$d5 === void 0 ? void 0 : _endDatePickerProps$d5.disabled]);
  const openEndDatePicker = () => {
    setIsFromVisible(false);
    setIsToVisible(true);
  };
  const closeEndDatePicker = () => {
    setIsToVisible(false);
  };
  const handleTimePickerKeyPress = e => {
    var _props$startDatePicke, _props$startDatePicke2;
    if (e.key === 'Enter') {
      openEndDatePicker();
    }
    (_props$startDatePicke = props.startDatePickerProps) === null || _props$startDatePicke === void 0 || (_props$startDatePicke = _props$startDatePicke.timeInputProps) === null || _props$startDatePicke === void 0 || (_props$startDatePicke2 = _props$startDatePicke.onKeyDown) === null || _props$startDatePicke2 === void 0 || _props$startDatePicke2.call(_props$startDatePicke, e);
  };
  return jsxs("div", {
    className: `${classNamePrefix}-rangepicker`,
    ...wrapperProps,
    "data-focused": isRangeInputFocused,
    css: /*#__PURE__*/css({
      display: 'flex',
      alignItems: 'center',
      minWidth,
      maxWidth,
      width
    }, process.env.NODE_ENV === "production" ? "" : ";label:RangePicker;"),
    children: [jsx(DatePicker, {
      quickActions: quickActionsWithHandler,
      prefix: "Start:",
      open: isFromVisible,
      onOpenChange: setIsFromVisible,
      okButtonLabel: "Next",
      ...startDatePickerProps,
      id: id,
      ref: fromInputRef,
      disabled: disabled || (startDatePickerProps === null || startDatePickerProps === void 0 ? void 0 : startDatePickerProps.disabled),
      onChange: e => handleUpdateDate(e, true),
      includeTime: includeTime,
      allowClear: allowClear,
      datePickerProps: {
        ...(startDatePickerProps === null || startDatePickerProps === void 0 ? void 0 : startDatePickerProps.datePickerProps),
        disabled: disabledDates.startDisabled
      },
      timeInputProps: {
        onKeyDown: handleTimePickerKeyPress
      }
      // @ts-expect-error - DatePickerProps does not have a mode property in the public API but is needed for this use case
      ,
      mode: "range",
      selected: range,
      value: range === null || range === void 0 ? void 0 : range.from,
      dateTimeDisabledFn: date => checkIfDateTimeIsDisabled(date, true),
      onFocus: e => {
        var _startDatePickerProps7;
        setIsRangeInputFocused(true);
        startDatePickerProps === null || startDatePickerProps === void 0 || (_startDatePickerProps7 = startDatePickerProps.onFocus) === null || _startDatePickerProps7 === void 0 || _startDatePickerProps7.call(startDatePickerProps, e);
      },
      onBlur: e => {
        var _startDatePickerProps8;
        setIsRangeInputFocused(false);
        startDatePickerProps === null || startDatePickerProps === void 0 || (_startDatePickerProps8 = startDatePickerProps.onBlur) === null || _startDatePickerProps8 === void 0 || _startDatePickerProps8.call(startDatePickerProps, e);
      },
      css: _ref3,
      wrapperProps: {
        style: {
          width: '50%'
        }
      },
      onOkPress: openEndDatePicker
    }), jsx(DatePicker, {
      quickActions: quickActionsWithHandler,
      prefix: "End:",
      min: range === null || range === void 0 || (_range$from = range.from) === null || _range$from === void 0 ? void 0 : _range$from.toString(),
      okButtonLabel: "Close",
      ...endDatePickerProps,
      ref: toInputRef,
      disabled: disabled || (endDatePickerProps === null || endDatePickerProps === void 0 ? void 0 : endDatePickerProps.disabled),
      onChange: e => handleUpdateDate(e, false),
      includeTime: includeTime,
      open: isToVisible,
      onOpenChange: setIsToVisible,
      allowClear: allowClear,
      datePickerProps: {
        ...(endDatePickerProps === null || endDatePickerProps === void 0 ? void 0 : endDatePickerProps.datePickerProps),
        disabled: disabledDates.endDisabled
      }
      // @ts-expect-error - DatePickerProps does not have a mode property in the public API but is needed for this use case
      ,
      mode: "range",
      selected: range,
      value: range === null || range === void 0 ? void 0 : range.to,
      dateTimeDisabledFn: date => checkIfDateTimeIsDisabled(date, false),
      onFocus: e => {
        var _startDatePickerProps9;
        setIsRangeInputFocused(true);
        startDatePickerProps === null || startDatePickerProps === void 0 || (_startDatePickerProps9 = startDatePickerProps.onFocus) === null || _startDatePickerProps9 === void 0 || _startDatePickerProps9.call(startDatePickerProps, e);
      },
      onBlur: e => {
        var _startDatePickerProps10;
        setIsRangeInputFocused(false);
        startDatePickerProps === null || startDatePickerProps === void 0 || (_startDatePickerProps10 = startDatePickerProps.onBlur) === null || _startDatePickerProps10 === void 0 || _startDatePickerProps10.call(startDatePickerProps, e);
      },
      css: /*#__PURE__*/css({
        borderTopLeftRadius: 0,
        borderBottomLeftRadius: 0,
        left: -1
      }, process.env.NODE_ENV === "production" ? "" : ";label:RangePicker;"),
      wrapperProps: {
        style: {
          width: '50%'
        }
      },
      onOkPress: closeEndDatePicker
    })]
  });
};

const RadioGroupContext = /*#__PURE__*/React__default.createContext('medium');
const Root$3 = /*#__PURE__*/React__default.forwardRef((_ref, forwardedRef) => {
  let {
    size,
    componentId,
    analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange],
    valueHasNoPii,
    onValueChange,
    ...props
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  const contextValue = React__default.useMemo(() => size !== null && size !== void 0 ? size : 'medium', [size]);
  const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.PillControl,
    componentId,
    analyticsEvents: memoizedAnalyticsEvents,
    valueHasNoPii
  });
  const onValueChangeWrapper = useCallback(value => {
    var _eventContext$onValue;
    (_eventContext$onValue = eventContext.onValueChange) === null || _eventContext$onValue === void 0 || _eventContext$onValue.call(eventContext, value);
    onValueChange === null || onValueChange === void 0 || onValueChange(value);
  }, [eventContext, onValueChange]);
  return jsx(RadioGroupContext.Provider, {
    value: contextValue,
    children: jsx(RadioGroup, {
      css: /*#__PURE__*/css({
        display: 'flex',
        flexWrap: 'wrap',
        gap: theme.spacing.sm
      }, process.env.NODE_ENV === "production" ? "" : ";label:Root;"),
      onValueChange: onValueChangeWrapper,
      ...props,
      ref: forwardedRef
    })
  });
});
const Item = /*#__PURE__*/React__default.forwardRef((_ref2, forwardedRef) => {
  let {
    children,
    icon,
    ...props
  } = _ref2;
  const size = React__default.useContext(RadioGroupContext);
  const {
    theme
  } = useDesignSystemTheme();
  const iconClass = 'pill-control-icon';
  const css$1 = useRadioGroupItemStyles(size, iconClass);
  return jsxs(RadioGroupItem, {
    css: css$1,
    ...props,
    children: [icon && jsx("span", {
      className: iconClass,
      css: /*#__PURE__*/css({
        marginRight: size === 'large' ? theme.spacing.sm : theme.spacing.xs,
        [`& > .anticon`]: {
          verticalAlign: `-3px`
        }
      }, process.env.NODE_ENV === "production" ? "" : ";label:Item;"),
      children: icon
    }), children]
  });
});
const useRadioGroupItemStyles = (size, iconClass) => {
  const {
    theme
  } = useDesignSystemTheme();
  return {
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    appearance: 'none',
    textDecoration: 'none',
    background: 'none',
    border: '1px solid',
    cursor: 'pointer',
    backgroundColor: theme.colors.actionDefaultBackgroundDefault,
    borderColor: theme.colors.border,
    color: theme.colors.textPrimary,
    lineHeight: theme.typography.lineHeightBase,
    height: 32,
    paddingInline: '12px',
    fontWeight: theme.typography.typographyRegularFontWeight,
    fontSize: theme.typography.fontSizeBase,
    borderRadius: theme.spacing.md,
    transition: 'background-color 0.2s ease-in-out, border-color 0.2s ease-in-out',
    [`& > .${iconClass}`]: {
      color: theme.colors.textSecondary,
      ...(size === 'large' ? {
        backgroundColor: theme.colors.tagDefault,
        padding: theme.spacing.sm,
        borderRadius: theme.spacing.md
      } : {})
    },
    '&[data-state="checked"]': {
      backgroundColor: theme.colors.actionDefaultBackgroundPress,
      borderColor: 'transparent',
      color: theme.colors.textPrimary,
      // outline
      outlineStyle: 'solid',
      outlineWidth: '2px',
      outlineOffset: '0px',
      outlineColor: theme.isDarkMode ? theme.colors.actionLinkDefault : theme.colors.actionLinkDefault,
      '&:hover': {
        backgroundColor: theme.colors.actionDefaultBackgroundPress,
        borderColor: theme.colors.actionLinkPress,
        color: 'inherit'
      },
      [`& > .${iconClass}, &:hover > .${iconClass}`]: {
        color: theme.colors.actionDefaultTextPress,
        ...(size === 'large' ? {
          backgroundColor: theme.colors.actionIconBackgroundPress
        } : {})
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
        ...(size === 'large' ? {
          backgroundColor: theme.colors.actionIconBackgroundHover
        } : {})
      }
    },
    '&:active': {
      backgroundColor: theme.colors.actionDefaultBackgroundPress,
      borderColor: theme.colors.actionLinkPress,
      color: theme.colors.actionDefaultTextPress,
      [`& > .${iconClass}`]: {
        color: 'inherit',
        ...(size === 'large' ? {
          backgroundColor: theme.colors.actionIconBackgroundPress
        } : {})
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
    ...(size === 'small' ? {
      height: 24,
      lineHeight: theme.typography.lineHeightSm,
      paddingInline: theme.spacing.sm
    } : {}),
    ...(size === 'large' ? {
      height: 44,
      lineHeight: theme.typography.lineHeightXl,
      paddingInline: theme.spacing.md,
      paddingInlineStart: '6px',
      borderRadius: theme.spacing.lg
    } : {})
  };
};

var PillControl = /*#__PURE__*/Object.freeze({
  __proto__: null,
  Item: Item,
  Root: Root$3
});

const PreviewCard = _ref => {
  let {
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
    ...props
  } = _ref;
  const styles = usePreviewCardStyles({
    onClick,
    size
  });
  const tabIndex = onClick ? 0 : undefined;
  const role = onClick ? 'button' : undefined;
  const showFooter = startActions || endActions;
  const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.PreviewCard,
    componentId,
    analyticsEvents: memoizedAnalyticsEvents
  });
  const onClickWrapper = useCallback(e => {
    if (onClick) {
      eventContext.onClick(e);
      onClick(e);
    }
  }, [eventContext, onClick]);
  return jsxs("div", {
    ...addDebugOutlineIfEnabled(),
    css: [styles['container'], dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:PreviewCard;"],
    tabIndex: tabIndex,
    onClick: onClickWrapper,
    role: role,
    ...props,
    children: [image && jsx("div", {
      css: styles['image'],
      children: image
    }), jsxs("div", {
      css: styles['header'],
      children: [icon && jsx("div", {
        children: icon
      }), jsxs("div", {
        css: styles['titleWrapper'],
        children: [title && jsx("div", {
          css: styles['title'],
          children: title
        }), subtitle && jsx("div", {
          css: styles['subTitle'],
          children: subtitle
        })]
      }), titleActions && jsx("div", {
        children: titleActions
      })]
    }), children && jsx("div", {
      css: styles['childrenWrapper'],
      children: children
    }), showFooter && jsxs("div", {
      css: styles['footer'],
      children: [jsx("div", {
        css: styles['action'],
        children: startActions
      }), jsx("div", {
        css: styles['action'],
        children: endActions
      })]
    })]
  });
};
const usePreviewCardStyles = _ref2 => {
  let {
    onClick,
    size
  } = _ref2;
  const {
    theme
  } = useDesignSystemTheme();
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
      cursor: isInteractive ? 'pointer' : 'default',
      ...(isInteractive && {
        '&:hover, &:focus-within': {
          boxShadow: theme.general.shadowLow
        },
        '&:active': {
          background: theme.colors.actionTertiaryBackgroundPress,
          borderColor: theme.colors.actionDefaultBorderHover,
          boxShadow: theme.general.shadowLow
        },
        '&:focus': {
          outlineColor: theme.colors.actionPrimaryBackgroundDefault,
          outlineWidth: 2,
          outlineOffset: -2,
          outlineStyle: 'solid',
          boxShadow: theme.general.shadowLow,
          borderColor: theme.colors.actionDefaultBorderHover
        },
        '&:active:not(:focus):not(:focus-within)': {
          background: 'transparent',
          borderColor: theme.colors.border
        }
      })
    },
    image: {
      '& > *': {
        borderRadius: theme.legacyBorders.borderRadiusMd
      }
    },
    header: {
      display: 'flex',
      alignItems: 'center',
      gap: theme.spacing.sm
    },
    title: {
      fontWeight: theme.typography.typographyBoldFontWeight,
      color: theme.colors.textPrimary,
      lineHeight: theme.typography.lineHeightSm
    },
    subTitle: {
      lineHeight: theme.typography.lineHeightSm
    },
    titleWrapper: {
      flexGrow: 1,
      overflow: 'hidden'
    },
    childrenWrapper: {
      flexGrow: 1
    },
    footer: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      flexWrap: 'wrap'
    },
    action: {
      overflow: 'hidden',
      // to ensure focus ring is rendered
      margin: theme.spacing.md * -1,
      padding: theme.spacing.md
    }
  };
};

const progressContextDefaults = {
  progress: 0
};
const ProgressContext = /*#__PURE__*/createContext(progressContextDefaults);
const ProgressContextProvider = _ref => {
  let {
    children,
    value
  } = _ref;
  return jsx(ProgressContext.Provider, {
    value: value,
    children: children
  });
};

const getProgressRootStyles = (theme, _ref) => {
  let {
    minWidth,
    maxWidth
  } = _ref;
  const styles = {
    position: 'relative',
    overflow: 'hidden',
    backgroundColor: theme.colors.progressTrack,
    height: theme.spacing.sm,
    width: '100%',
    borderRadius: theme.spacing.xs,
    ...(minWidth && {
      minWidth
    }),
    ...(maxWidth && {
      maxWidth
    }),
    /* Fix overflow clipping in Safari */
    /* https://gist.github.com/domske/b66047671c780a238b51c51ffde8d3a0 */
    transform: 'translateZ(0)'
  };
  return /*#__PURE__*/css(importantify(styles), process.env.NODE_ENV === "production" ? "" : ";label:getProgressRootStyles;");
};
const Root$2 = props => {
  const {
    children,
    value,
    minWidth,
    maxWidth,
    ...restProps
  } = props;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(ProgressContextProvider, {
    value: {
      progress: value
    },
    children: jsx(Progress$1.Root, {
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
const getProgressIndicatorStyles = theme => {
  const styles = {
    backgroundColor: theme.colors.progressFill,
    height: '100%',
    width: '100%',
    transition: 'transform 300ms linear',
    borderRadius: theme.spacing.xs
  };
  return /*#__PURE__*/css(importantify(styles), process.env.NODE_ENV === "production" ? "" : ";label:getProgressIndicatorStyles;");
};
const Indicator = props => {
  const {
    progress
  } = React__default.useContext(ProgressContext);
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(Progress$1.Indicator, {
    css: getProgressIndicatorStyles(theme),
    style: {
      transform: `translateX(-${100 - (progress !== null && progress !== void 0 ? progress : 100)}%)`
    },
    ...props
  });
};

var Progress = /*#__PURE__*/Object.freeze({
  __proto__: null,
  Indicator: Indicator,
  Root: Root$2
});

function _EMOTION_STRINGIFIED_CSS_ERROR__() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
var _ref = process.env.NODE_ENV === "production" ? {
  name: "1tciq3q",
  styles: "position:relative;display:flex;align-items:center;&[data-orientation=\"vertical\"]{flex-direction:column;width:20px;height:100px;}&[data-orientation=\"horizontal\"]{height:20px;width:200px;}"
} : {
  name: "18im58f-getRootStyles",
  styles: "position:relative;display:flex;align-items:center;&[data-orientation=\"vertical\"]{flex-direction:column;width:20px;height:100px;}&[data-orientation=\"horizontal\"]{height:20px;width:200px;};label:getRootStyles;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__
};
const getRootStyles$1 = () => {
  return _ref;
};
const Root$1 = /*#__PURE__*/forwardRef((props, ref) => {
  return jsx(RadixSlider.Root, {
    ...addDebugOutlineIfEnabled(),
    css: getRootStyles$1(),
    ...props,
    ref: ref
  });
});
const getTrackStyles = theme => {
  return /*#__PURE__*/css({
    backgroundColor: theme.colors.grey100,
    position: 'relative',
    flexGrow: 1,
    borderRadius: 9999,
    '&[data-orientation="vertical"]': {
      width: 3
    },
    '&[data-orientation="horizontal"]': {
      height: 3
    }
  }, process.env.NODE_ENV === "production" ? "" : ";label:getTrackStyles;");
};
const Track = /*#__PURE__*/forwardRef((props, ref) => {
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(RadixSlider.Track, {
    css: getTrackStyles(theme),
    ...props,
    ref: ref
  });
});
const getRangeStyles = theme => {
  return /*#__PURE__*/css({
    backgroundColor: theme.colors.primary,
    position: 'absolute',
    borderRadius: 9999,
    height: '100%',
    '&[data-disabled]': {
      backgroundColor: theme.colors.grey100
    }
  }, process.env.NODE_ENV === "production" ? "" : ";label:getRangeStyles;");
};
const Range = /*#__PURE__*/forwardRef((props, ref) => {
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(RadixSlider.Range, {
    css: getRangeStyles(theme),
    ...props,
    ref: ref
  });
});
const getThumbStyles = theme => {
  return /*#__PURE__*/css({
    display: 'block',
    width: 20,
    height: 20,
    backgroundColor: theme.colors.actionPrimaryBackgroundDefault,
    boxShadow: `0 2px 4px 0 ${theme.colors.grey400}`,
    borderRadius: 10,
    outline: 'none',
    '&:hover': {
      backgroundColor: theme.colors.actionPrimaryBackgroundHover
    },
    '&:focus': {
      backgroundColor: theme.colors.actionPrimaryBackgroundPress
    },
    '&[data-disabled]': {
      backgroundColor: theme.colors.grey200,
      boxShadow: 'none'
    }
  }, process.env.NODE_ENV === "production" ? "" : ";label:getThumbStyles;");
};
const Thumb = /*#__PURE__*/forwardRef((props, ref) => {
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(RadixSlider.Thumb, {
    css: getThumbStyles(theme),
    ...props,
    ref: ref
  });
});

var Slider = /*#__PURE__*/Object.freeze({
  __proto__: null,
  Range: Range,
  Root: Root$1,
  Thumb: Thumb,
  Track: Track
});

const getRootStyles = theme => {
  return /*#__PURE__*/css({
    alignItems: 'center',
    backgroundColor: theme.colors.backgroundSecondary,
    border: `1px solid ${theme.colors.borderDecorative}`,
    borderRadius: theme.legacyBorders.borderRadiusMd,
    boxShadow: theme.general.shadowLow,
    display: 'flex',
    gap: theme.spacing.md,
    width: 'max-content',
    padding: theme.spacing.sm
  }, process.env.NODE_ENV === "production" ? "" : ";label:getRootStyles;");
};
const Root = /*#__PURE__*/forwardRef((props, ref) => {
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(RadixToolbar.Root, {
    ...addDebugOutlineIfEnabled(),
    css: getRootStyles(theme),
    ...props,
    ref: ref
  });
});
const Button = /*#__PURE__*/forwardRef((props, ref) => {
  return jsx(RadixToolbar.Button, {
    ...props,
    ref: ref
  });
});
const getSeparatorStyles = theme => {
  return /*#__PURE__*/css({
    alignSelf: 'stretch',
    backgroundColor: theme.colors.borderDecorative,
    width: 1
  }, process.env.NODE_ENV === "production" ? "" : ";label:getSeparatorStyles;");
};
const Separator = /*#__PURE__*/forwardRef((props, ref) => {
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(RadixToolbar.Separator, {
    css: getSeparatorStyles(theme),
    ...props,
    ref: ref
  });
});
const Link = /*#__PURE__*/forwardRef((props, ref) => {
  return jsx(RadixToolbar.Link, {
    ...props,
    ref: ref
  });
});
const ToggleGroup = /*#__PURE__*/forwardRef((props, ref) => {
  return jsx(RadixToolbar.ToggleGroup, {
    ...props,
    ref: ref
  });
});
const getToggleItemStyles = theme => {
  return /*#__PURE__*/css({
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
  }, process.env.NODE_ENV === "production" ? "" : ";label:getToggleItemStyles;");
};
const ToggleItem = /*#__PURE__*/forwardRef((props, ref) => {
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(RadixToolbar.ToggleItem, {
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

const flattenData = function (data, expandedRows) {
  let depth = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : 0;
  let parentId = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : null;
  return data.reduce((acc, node) => {
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
function treeGridReducer(state, action) {
  switch (action.type) {
    case 'TOGGLE_ROW_EXPANDED':
      return {
        ...state,
        expandedRows: {
          ...state.expandedRows,
          [action.rowId]: !state.expandedRows[action.rowId]
        }
      };
    case 'SET_ACTIVE_ROW':
      return {
        ...state,
        activeRowIndex: action.rowIndex
      };
    default:
      return state;
  }
}
const findFocusableElementForCellIndex = (row, cellIndex) => {
  const cell = row.cells[cellIndex];
  return (cell === null || cell === void 0 ? void 0 : cell.querySelector('[tabindex], button, a, input, select, textarea')) || null;
};
const findNextFocusableCellIndexInRow = (row, columns, startIndex, direction) => {
  const cells = Array.from(row.cells);
  const increment = direction === 'next' ? 1 : -1;
  const limit = direction === 'next' ? cells.length : -1;
  for (let i = startIndex + increment; i !== limit; i += increment) {
    var _cell$textContent;
    const cell = cells[i];
    const column = columns[i];
    const focusableElement = findFocusableElementForCellIndex(row, i);
    const cellContent = cell === null || cell === void 0 || (_cell$textContent = cell.textContent) === null || _cell$textContent === void 0 ? void 0 : _cell$textContent.trim();
    if (focusableElement || !column.contentFocusable && cellContent) {
      return i;
    }
  }
  return -1;
};
const TreeGrid = _ref => {
  let {
    data,
    columns,
    renderCell,
    renderRow,
    renderTable,
    renderHeader,
    onRowKeyboardSelect,
    onCellKeyboardSelect,
    includeHeader = false,
    initialState = {
      expandedRows: {}
    }
  } = _ref;
  const [state, dispatch] = useReducer(treeGridReducer, {
    ...initialState,
    activeRowIndex: 0
  });
  const gridRef = useRef(null);
  const flattenedData = useMemo(() => flattenData(data, state.expandedRows), [data, state.expandedRows]);
  const toggleRowExpanded = useCallback(rowId => {
    dispatch({
      type: 'TOGGLE_ROW_EXPANDED',
      rowId
    });
  }, []);
  const focusRow = useCallback(rowIndex => {
    var _gridRef$current;
    const row = (_gridRef$current = gridRef.current) === null || _gridRef$current === void 0 ? void 0 : _gridRef$current.querySelector(`tbody tr:nth-child(${rowIndex + 1})`);
    row === null || row === void 0 || row.focus();
    dispatch({
      type: 'SET_ACTIVE_ROW',
      rowIndex
    });
  }, []);
  const handleKeyDown = useCallback((event, rowIndex) => {
    const {
      key
    } = event;
    let newRowIndex = rowIndex;
    const closestTd = event.target.closest('td');
    if (!gridRef.current || !gridRef.current.contains(document.activeElement)) {
      return;
    }
    const handleArrowVerticalNavigation = direction => {
      if (closestTd) {
        var _closestTd$closest;
        const currentCellIndex = closestTd.cellIndex;
        let targetRow = (_closestTd$closest = closestTd.closest('tr')) === null || _closestTd$closest === void 0 ? void 0 : _closestTd$closest[`${direction}ElementSibling`];
        const moveFocusToRow = row => {
          var _row$cells$currentCel;
          const focusableElement = findFocusableElementForCellIndex(row, currentCellIndex);
          const cellContent = (_row$cells$currentCel = row.cells[currentCellIndex]) === null || _row$cells$currentCel === void 0 || (_row$cells$currentCel = _row$cells$currentCel.textContent) === null || _row$cells$currentCel === void 0 ? void 0 : _row$cells$currentCel.trim();
          if (focusableElement || !columns[currentCellIndex].contentFocusable && cellContent) {
            event.preventDefault();
            focusElement(focusableElement || row.cells[currentCellIndex], flattenedData.findIndex(r => r.id === row.dataset['id']));
            return true;
          }
          return false;
        };
        while (targetRow) {
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
    const handleArrowHorizontalNavigation = direction => {
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
            if (!state.expandedRows[flattenedData[rowIndex].id]) {
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
          if (state.expandedRows[flattenedData[rowIndex].id]) {
            toggleRowExpanded(flattenedData[rowIndex].id);
          } else if (flattenedData[rowIndex].depth && flattenedData[rowIndex].depth > 0) {
            newRowIndex = flattenedData.findIndex(row => row.id === flattenedData[rowIndex].parentId);
          }
        }
        return;
      }

      // If we're at the edge of the row, handle expanding/collapsing or moving to parent/child
      if (direction === 'next') {
        if (flattenedData[rowIndex].children && !state.expandedRows[flattenedData[rowIndex].id]) {
          toggleRowExpanded(flattenedData[rowIndex].id);
        }
      } else {
        if (state.expandedRows[flattenedData[rowIndex].id]) {
          toggleRowExpanded(flattenedData[rowIndex].id);
        } else if (flattenedData[rowIndex].depth && flattenedData[rowIndex].depth > 0) {
          newRowIndex = flattenedData.findIndex(row => row.id === flattenedData[rowIndex].parentId);
        }
      }
    };
    const handleEnterKey = () => {
      if (closestTd) {
        onCellKeyboardSelect === null || onCellKeyboardSelect === void 0 || onCellKeyboardSelect(flattenedData[rowIndex].id, columns[closestTd.cellIndex].id);
      } else if (document.activeElement instanceof HTMLTableRowElement) {
        onRowKeyboardSelect === null || onRowKeyboardSelect === void 0 || onRowKeyboardSelect(flattenedData[rowIndex].id);
      }
    };
    switch (key) {
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
      focusRow(newRowIndex);
    }
  }, [state.expandedRows, columns, flattenedData, toggleRowExpanded, onRowKeyboardSelect, onCellKeyboardSelect, focusRow]);
  const focusElement = (element, rowIndex) => {
    if (element) {
      element.focus();
      dispatch({
        type: 'SET_ACTIVE_ROW',
        rowIndex
      });
    }
  };
  const defaultRenderRow = useCallback(_ref2 => {
    let {
      rowProps,
      children
    } = _ref2;
    return jsx("tr", {
      ...rowProps,
      children: children
    });
  }, []);
  const defaultRenderTable = useCallback(_ref3 => {
    let {
      tableProps,
      children
    } = _ref3;
    return jsx("table", {
      ...tableProps,
      children: children
    });
  }, []);
  const defaultRenderHeader = useCallback(_ref4 => {
    let {
      columns,
      headerProps
    } = _ref4;
    return jsx("thead", {
      ...headerProps,
      children: jsx("tr", {
        children: columns.map(column => jsx("th", {
          role: "columnheader",
          children: column.header
        }, column.id))
      })
    });
  }, []);
  const renderRowWrapper = useCallback((row, rowIndex) => {
    const isExpanded = state.expandedRows[row.id];
    const isKeyboardActive = rowIndex === state.activeRowIndex;
    const rowProps = {
      key: row.id,
      'data-id': row.id,
      role: 'row',
      'aria-selected': false,
      'aria-level': (row.depth || 0) + 1,
      'aria-expanded': row.children ? isExpanded ? 'true' : 'false' : undefined,
      tabIndex: isKeyboardActive ? 0 : -1,
      onKeyDown: e => handleKeyDown(e, rowIndex)
    };
    const children = columns.map((column, colIndex) => {
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
  }, [state.activeRowIndex, state.expandedRows, handleKeyDown, renderCell, renderRow, defaultRenderRow, toggleRowExpanded, columns]);
  return (renderTable || defaultRenderTable)({
    tableProps: {
      role: 'treegrid',
      ref: gridRef
    },
    children: jsxs(Fragment, {
      children: [includeHeader && (renderHeader || defaultRenderHeader)({
        columns,
        headerProps: {}
      }), jsx("tbody", {
        children: flattenedData.map((row, index) => renderRowWrapper(row, index))
      })]
    })
  });
};

export { BANNER_MAX_HEIGHT, BANNER_MIN_HEIGHT, Banner, DatePicker, PillControl, PreviewCard, Progress, RangePicker, Slider, Toolbar, TreeGrid, getDatePickerQuickActionBasic, getRangeQuickActionsBasic };
//# sourceMappingURL=development.js.map
