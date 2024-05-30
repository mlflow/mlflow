import { css } from '@emotion/react';
import React__default, { forwardRef, useCallback } from 'react';
import { W as WarningIcon, q as DangerIcon, u as useDesignSystemTheme, b as useDesignSystemEventComponentCallbacks, c as DesignSystemEventProviderComponentTypes, d as DesignSystemEventProviderAnalyticsEventTypes, e as useNotifyOnFirstView, B as Button$1, C as CloseIcon, f as addDebugOutlineIfEnabled, T as Typography, a0 as primitiveColors, p as getShadowScrollStyles } from './Typography-78b12af3.js';
import { jsx, Fragment, jsxs } from '@emotion/react/jsx-runtime';
import { M as MegaphoneIcon, P as PlusIcon, C as CloseSmallIcon } from './PlusIcon-e78c4843.js';
import * as RadixSlider from '@radix-ui/react-slider';
import * as RadixToolbar from '@radix-ui/react-toolbar';
export { S as Stepper } from './Stepper-2c82de4e.js';
import { RadioGroup, RadioGroupItem } from '@radix-ui/react-radio-group';
import { useMergeRefs } from '@floating-ui/react';
import * as ScrollArea from '@radix-ui/react-scroll-area';
import * as RadixTabs from '@radix-ui/react-tabs';
import 'antd';
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

function _EMOTION_STRINGIFIED_CSS_ERROR__$1() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
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
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$1
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
    info: {
      backgroundDefaultColor: theme.colors.actionPrimaryBackgroundDefault,
      actionButtonBackgroundHoverColor: theme.colors.actionPrimaryBackgroundHover,
      actionButtonBackgroundPressColor: theme.colors.actionPrimaryBackgroundPress,
      textColor: theme.colors.actionPrimaryTextDefault,
      textHoverColor: theme.colors.actionPrimaryTextHover,
      textPressColor: theme.colors.actionPrimaryTextPress,
      borderDefaultColor: theme.colors.actionPrimaryBackgroundDefault
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
    analyticsEvents
  } = props;
  const [closed, setClosed] = React__default.useState(false);
  const {
    theme
  } = useDesignSystemTheme();
  const styles = useStyles(props, theme);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.Banner,
    componentId,
    analyticsEvents: analyticsEvents !== null && analyticsEvents !== void 0 ? analyticsEvents : [DesignSystemEventProviderAnalyticsEventTypes.OnView]
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
const Root$3 = /*#__PURE__*/forwardRef((props, ref) => {
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
  Root: Root$3,
  Thumb: Thumb,
  Track: Track
});

const getRootStyles = theme => {
  return /*#__PURE__*/css({
    alignItems: 'center',
    backgroundColor: theme.colors.backgroundSecondary,
    border: `1px solid ${theme.colors.borderDecorative}`,
    borderRadius: theme.borders.borderRadiusMd,
    boxShadow: theme.general.shadowLow,
    display: 'flex',
    gap: theme.spacing.md,
    width: 'max-content',
    padding: theme.spacing.sm
  }, process.env.NODE_ENV === "production" ? "" : ";label:getRootStyles;");
};
const Root$2 = /*#__PURE__*/forwardRef((props, ref) => {
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
  Root: Root$2,
  Separator: Separator,
  ToggleGroup: ToggleGroup,
  ToggleItem: ToggleItem
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
    ...props
  } = _ref;
  const styles = usePreviewCardStyles({
    onClick,
    size
  });
  const tabIndex = onClick ? 0 : undefined;
  const role = onClick ? 'button' : undefined;
  const showFooter = startActions || endActions;
  return jsxs("div", {
    ...addDebugOutlineIfEnabled(),
    css: [styles['container'], dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:PreviewCard;"],
    tabIndex: tabIndex,
    onClick: onClick,
    role: role,
    ...props,
    children: [image && jsx("div", {
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
      borderRadius: theme.borders.borderRadiusLg,
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
      overflow: 'hidden'
    }
  };
};

const RadioGroupContext = /*#__PURE__*/React__default.createContext('medium');
const Root$1 = /*#__PURE__*/React__default.forwardRef((_ref, forwardedRef) => {
  let {
    size,
    componentId,
    analyticsEvents,
    valueHasNoPii,
    onValueChange,
    ...props
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  const contextValue = React__default.useMemo(() => size !== null && size !== void 0 ? size : 'medium', [size]);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.PillControl,
    componentId,
    analyticsEvents: analyticsEvents !== null && analyticsEvents !== void 0 ? analyticsEvents : [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange],
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
  Root: Root$1
});

const TabsV2RootContext = /*#__PURE__*/React__default.createContext({
  activeValue: undefined
});
const TabsV2ListContext = /*#__PURE__*/React__default.createContext({
  viewportRef: {
    current: null
  }
});
const Root = /*#__PURE__*/React__default.forwardRef((_ref, forwardedRef) => {
  let {
    value,
    defaultValue,
    onValueChange,
    ...props
  } = _ref;
  const isControlled = value !== undefined;
  const [uncontrolledActiveValue, setUncontrolledActiveValue] = React__default.useState(defaultValue);
  const onValueChangeWrapper = value => {
    if (onValueChange) {
      onValueChange(value);
    }
    if (!isControlled) {
      setUncontrolledActiveValue(value);
    }
  };
  return jsx(TabsV2RootContext.Provider, {
    value: {
      activeValue: isControlled ? value : uncontrolledActiveValue
    },
    children: jsx(RadixTabs.Root, {
      value: value,
      defaultValue: defaultValue,
      onValueChange: onValueChangeWrapper,
      ...props,
      ref: forwardedRef
    })
  });
});
const List = /*#__PURE__*/React__default.forwardRef((_ref2, forwardedRef) => {
  let {
    addButtonProps,
    scrollAreaViewportCss,
    children,
    dangerouslyAppendEmotionCSS,
    ...props
  } = _ref2;
  const viewportRef = React__default.useRef(null);
  const css = useListStyles();
  return jsx(TabsV2ListContext.Provider, {
    value: {
      viewportRef
    },
    children: jsxs("div", {
      css: [css['container'], dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:List;"],
      children: [jsxs(ScrollArea.Root, {
        type: "hover",
        css: [css['root'], process.env.NODE_ENV === "production" ? "" : ";label:List;"],
        children: [jsx(ScrollArea.Viewport, {
          css: [css['viewport'], scrollAreaViewportCss, process.env.NODE_ENV === "production" ? "" : ";label:List;"],
          ref: viewportRef,
          children: jsx(RadixTabs.List, {
            css: css['list'],
            ...props,
            ref: forwardedRef,
            children: children
          })
        }), jsx(ScrollArea.Scrollbar, {
          orientation: "horizontal",
          css: css['scrollbar'],
          children: jsx(ScrollArea.Thumb, {
            css: css['thumb']
          })
        })]
      }), addButtonProps && jsx("div", {
        css: [css['addButtonContainer'], addButtonProps.dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:List;"],
        children: jsx(Button$1, {
          icon: jsx(PlusIcon, {}),
          size: "small",
          "aria-label": "Add tab",
          css: css['addButton'],
          onClick: addButtonProps.onClick,
          componentId: addButtonProps.componentId,
          className: addButtonProps.className
        })
      })]
    })
  });
});
const Trigger = /*#__PURE__*/React__default.forwardRef((_ref3, forwardedRef) => {
  let {
    onClose,
    value,
    disabled,
    children,
    ...props
  } = _ref3;
  const triggerRef = React__default.useRef(null);
  const mergedRef = useMergeRefs([forwardedRef, triggerRef]);
  const {
    activeValue
  } = React__default.useContext(TabsV2RootContext);
  const {
    viewportRef
  } = React__default.useContext(TabsV2ListContext);
  const isClosable = onClose !== undefined && !disabled;
  const css = useTriggerStyles();
  React__default.useEffect(() => {
    if (triggerRef.current && viewportRef.current && activeValue === value) {
      const viewportPosition = viewportRef.current.getBoundingClientRect();
      const triggerPosition = triggerRef.current.getBoundingClientRect();
      if (triggerPosition.left < viewportPosition.left) {
        viewportRef.current.scrollLeft -= viewportPosition.left - triggerPosition.left;
      } else if (triggerPosition.right > viewportPosition.right) {
        viewportRef.current.scrollLeft += triggerPosition.right - viewportPosition.right;
      }
    }
  }, [viewportRef, triggerRef, activeValue, value]);
  return jsxs(RadixTabs.Trigger, {
    css: css['trigger'],
    value: value,
    disabled: disabled
    // The close icon cannot be focused within the trigger button
    // Instead, we close the tab when the Delete key is pressed
    ,
    onKeyDown: e => {
      if (onClose && e.key === 'Delete') {
        e.stopPropagation();
        e.preventDefault();
        onClose(value);
      }
    },
    ...props,
    ref: mergedRef,
    children: [children, isClosable &&
    // An icon is used instead of a button to prevent nesting a button within a button
    jsx(CloseSmallIcon, {
      onMouseDown: e => {
        // The Radix Tabs implementation only allows the trigger to be selected when the left mouse
        // button is clicked and not when the control key is pressed (to avoid MacOS right click).
        // Reimplementing the same behavior for the close icon in the trigger
        if (!disabled && e.button === 0 && e.ctrlKey === false) {
          // Clicking the close icon should not select the tab
          e.stopPropagation();
          e.preventDefault();
          onClose(value);
        }
      },
      css: css['closeSmallIcon'],
      "aria-hidden": "false",
      "aria-label": "Press delete to close the tab"
    })]
  });
});
const Content = /*#__PURE__*/React__default.forwardRef((_ref4, forwardedRef) => {
  let {
    ...props
  } = _ref4;
  const css = useContentStyles();
  return jsx(RadixTabs.Content, {
    css: css,
    ...props,
    ref: forwardedRef
  });
});
const useListStyles = () => {
  const {
    theme
  } = useDesignSystemTheme();
  return {
    container: {
      display: 'flex',
      borderBottom: `1px solid ${theme.colors.border}`,
      marginBottom: theme.spacing.md,
      height: theme.general.heightSm,
      boxSizing: 'border-box'
    },
    root: {
      overflow: 'hidden'
    },
    viewport: {
      ...getShadowScrollStyles(theme, {
        orientation: 'horizontal'
      })
    },
    list: {
      display: 'flex',
      alignItems: 'center'
    },
    scrollbar: {
      display: 'flex',
      flexDirection: 'column',
      userSelect: 'none',
      /* Disable browser handling of all panning and zooming gestures on touch devices */
      touchAction: 'none',
      height: 3
    },
    thumb: {
      flex: 1,
      background: theme.isDarkMode ? 'rgba(255, 255, 255, 0.2)' : 'rgba(17, 23, 28, 0.2)',
      '&:hover': {
        background: theme.isDarkMode ? 'rgba(255, 255, 255, 0.3)' : 'rgba(17, 23, 28, 0.3)'
      },
      borderRadius: theme.borders.borderRadiusMd,
      position: 'relative'
    },
    addButtonContainer: {
      flex: 1
    },
    addButton: {
      margin: '2px 0 6px 0'
    }
  };
};
const useTriggerStyles = () => {
  const {
    theme
  } = useDesignSystemTheme();
  return {
    trigger: {
      display: 'flex',
      alignItems: 'center',
      color: theme.colors.textSecondary,
      fontWeight: theme.typography.typographyBoldFontWeight,
      fontSize: theme.typography.fontSizeMd,
      lineHeight: theme.typography.lineHeightBase,
      backgroundColor: 'transparent',
      whiteSpace: 'nowrap',
      border: 'none',
      padding: `${theme.spacing.xs}px 0 ${theme.spacing.sm}px 0`,
      marginRight: theme.spacing.md,
      // The close icon is hidden on inactive tabs until the tab is hovered
      // Checking for the last icon to handle cases where the tab name includes an icon
      [`& > .anticon:last-of-type`]: {
        visibility: 'hidden'
      },
      '&:hover': {
        cursor: 'pointer',
        color: theme.colors.actionDefaultTextHover,
        [`& > .anticon:last-of-type`]: {
          visibility: 'visible'
        }
      },
      '&:active': {
        color: theme.colors.actionDefaultTextPress
      },
      outlineStyle: 'none',
      outlineColor: theme.colors.actionDefaultBorderFocus,
      '&:focus-visible': {
        outlineStyle: 'auto'
      },
      '&[data-state="active"]': {
        color: theme.colors.textPrimary,
        // Use box-shadow instead of border to prevent it from affecting the size of the element, which results in visual
        // jumping when switching tabs.
        boxShadow: `inset 0 -4px 0 ${theme.colors.actionPrimaryBackgroundDefault}`,
        // The close icon is always visible on active tabs
        [`& > .anticon:last-of-type`]: {
          visibility: 'visible'
        }
      },
      '&[data-disabled]': {
        color: theme.colors.actionDisabledText,
        '&:hover': {
          cursor: 'not-allowed'
        }
      }
    },
    closeSmallIcon: {
      marginLeft: theme.spacing.xs,
      color: theme.colors.textSecondary,
      '&:hover': {
        color: theme.colors.actionDefaultTextHover
      },
      '&:active': {
        color: theme.colors.actionDefaultTextPress
      }
    }
  };
};
const useContentStyles = () => {
  // This is needed so force mounted content is not displayed when the tab is inactive
  return {
    '&[data-state="inactive"]': {
      display: 'none'
    }
  };
};

var TabsV2 = /*#__PURE__*/Object.freeze({
  __proto__: null,
  Content: Content,
  List: List,
  Root: Root,
  Trigger: Trigger
});

export { BANNER_MAX_HEIGHT, BANNER_MIN_HEIGHT, Banner, PillControl, PreviewCard, Slider, TabsV2, Toolbar };
//# sourceMappingURL=development.js.map
