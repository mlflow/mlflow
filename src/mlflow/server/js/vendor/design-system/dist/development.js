import { css } from '@emotion/react';
import React__default, { useRef, useEffect, useState, forwardRef } from 'react';
import { Z as MegaphoneIcon, W as WarningIcon, k as DangerIcon, u as useDesignSystemTheme, G as useDesignSystemEventComponentCallbacks, E as DesignSystemEventProviderComponentTypes, F as DesignSystemEventProviderAnalyticsEventTypes, B as Button$1, C as CloseIcon, T as Typography, a4 as primitiveColors, f as Input, U as CursorIcon, Y as FaceSmileIcon, X as FaceNeutralIcon, V as FaceFrownIcon, e as CheckIcon, o as ChevronLeftIcon, b as ChevronRightIcon } from './Stepper-456afc57.js';
export { y as Stepper } from './Stepper-456afc57.js';
import { jsx, Fragment, jsxs } from '@emotion/react/jsx-runtime';
import * as RadixSlider from '@radix-ui/react-slider';
import * as RadixToolbar from '@radix-ui/react-toolbar';
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

// Define a module-global observer and a WeakMap on elements to hold callbacks
let sharedObserver = null;
const entryCallbackMap = new WeakMap();
const ensureSharedObserverExists = () => {
  if (!sharedObserver) {
    sharedObserver = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const entryCallback = entryCallbackMap.get(entry.target);
          if (entryCallback) {
            entryCallback();
            entryCallbackMap.delete(entry.target);
          }
        }
      });
    });
  }
};
function observeElement(element, callback) {
  var _sharedObserver;
  ensureSharedObserverExists();
  entryCallbackMap.set(element, callback);
  (_sharedObserver = sharedObserver) === null || _sharedObserver === void 0 || _sharedObserver.observe(element);
  return () => {
    if (element) {
      var _sharedObserver2;
      (_sharedObserver2 = sharedObserver) === null || _sharedObserver2 === void 0 || _sharedObserver2.unobserve(element);
      entryCallbackMap.delete(element);
    }
  };
}
/**
 * Checks if the element was viewed and calls the onView callback.
 * NOTE: This hook only triggers the onView callback once for the element.
 * @param onView - callback to be called when the element is viewed
 * @typeParam T - extends Element to specify the type of element being observed
 */
const useNotifyOnFirstView = _ref => {
  let {
    onView
  } = _ref;
  const isViewedRef = useRef(false);
  const elementRef = useRef(null);
  useEffect(() => {
    const element = elementRef.current;
    // If already viewed or element is not available, do nothing
    if (!element || isViewedRef.current) {
      return;
    }
    const callback = () => {
      isViewedRef.current = true;
      onView();
    };

    // If IntersectionObserver is not available, assume that the element is visible
    if (!window.IntersectionObserver) {
      callback();
      return;
    }
    return observeElement(element, callback);
  }, [onView]);
  return {
    elementRef
  };
};

function _EMOTION_STRINGIFIED_CSS_ERROR__$7() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
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
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$7
};
const useStyles = (props, theme) => {
  const bannerLevelToBannerColors = {
    info: {
      backgroundDefaultColor: theme.colors.actionPrimaryBackgroundDefault,
      actionButtonBackgroundHoverColor: theme.colors.actionPrimaryBackgroundHover,
      actionButtonBackgroundPressColor: theme.colors.actionPrimaryBackgroundPress,
      textColor: theme.colors.actionPrimaryTextDefault,
      textHoverColor: theme.colors.actionPrimaryTextHover,
      textPressColor: theme.colors.actionPrimaryTextPress
    },
    // TODO (PLAT-80558, zack.brody) Update hover and press states once we have colors for these
    warning: {
      backgroundDefaultColor: theme.colors.tagLemon,
      actionButtonBackgroundHoverColor: theme.colors.tagLemon,
      actionButtonBackgroundPressColor: theme.colors.tagLemon,
      textColor: primitiveColors.grey800,
      textHoverColor: primitiveColors.grey800,
      textPressColor: primitiveColors.grey800
    },
    error: {
      backgroundDefaultColor: theme.colors.actionDangerPrimaryBackgroundDefault,
      actionButtonBackgroundHoverColor: theme.colors.actionDangerPrimaryBackgroundHover,
      actionButtonBackgroundPressColor: theme.colors.actionDangerPrimaryBackgroundPress,
      textColor: theme.colors.actionPrimaryTextDefault,
      textHoverColor: theme.colors.actionPrimaryTextHover,
      textPressColor: theme.colors.actionPrimaryTextPress
    }
  };
  const colorScheme = bannerLevelToBannerColors[props.level];
  return {
    banner: /*#__PURE__*/css("max-height:", BANNER_MAX_HEIGHT, "px;display:flex;align-items:center;width:100%;padding:8px;box-sizing:border-box;background-color:", colorScheme.backgroundDefaultColor, ";" + (process.env.NODE_ENV === "production" ? "" : ";label:banner;")),
    iconContainer: /*#__PURE__*/css("display:flex;color:", colorScheme.textColor, ";align-self:", props.description ? 'flex-start' : 'center', ";box-sizing:border-box;max-width:60px;padding-top:4px;padding-bottom:4px;padding-right:", theme.spacing.xs, "px;" + (process.env.NODE_ENV === "production" ? "" : ";label:iconContainer;")),
    mainContent: /*#__PURE__*/css("flex-direction:column;align-self:", props.description ? 'flex-start' : 'center', ";display:flex;box-sizing:border-box;padding-right:", theme.spacing.sm, "px;padding-top:2px;padding-bottom:2px;min-width:", theme.spacing.lg, "px;width:100%;" + (process.env.NODE_ENV === "production" ? "" : ";label:mainContent;")),
    messageTextBlock: /*#__PURE__*/css("display:-webkit-box;-webkit-line-clamp:1;-webkit-box-orient:vertical;overflow:hidden;&&{color:", colorScheme.textColor, ";}" + (process.env.NODE_ENV === "production" ? "" : ";label:messageTextBlock;")),
    descriptionBlock: /*#__PURE__*/css("display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;&&{color:", colorScheme.textColor, ";}" + (process.env.NODE_ENV === "production" ? "" : ";label:descriptionBlock;")),
    rightContainer: _ref$1,
    closeIconContainer: /*#__PURE__*/css("display:flex;margin-left:", theme.spacing.xs, "px;box-sizing:border-box;line-height:0;" + (process.env.NODE_ENV === "production" ? "" : ";label:closeIconContainer;")),
    closeButton: /*#__PURE__*/css("cursor:pointer;background:none;border:none;margin:0;&&{height:24px!important;width:24px!important;padding:", theme.spacing.xs, "px!important;}&&:hover{background-color:transparent!important;}" + (process.env.NODE_ENV === "production" ? "" : ";label:closeButton;")),
    closeIcon: /*#__PURE__*/css("color:", colorScheme.textColor, "!important;" + (process.env.NODE_ENV === "production" ? "" : ";label:closeIcon;")),
    actionButtonContainer: /*#__PURE__*/css("margin-right:", theme.spacing.xs, "px;" + (process.env.NODE_ENV === "production" ? "" : ";label:actionButtonContainer;")),
    // Override design system colors to show the use the action text color for text and border.
    // Also overrides text for links.
    actionButton: /*#__PURE__*/css("color:", colorScheme.textColor, "!important;border-color:", colorScheme.textColor, "!important;&:focus,&:hover{border-color:", colorScheme.textHoverColor, "!important;color:", colorScheme.textHoverColor, "!important;background-color:", colorScheme.actionButtonBackgroundHoverColor, "!important;}&:active{border-color:", colorScheme.textPressColor, "!important;color:", colorScheme.textPressColor, "!important;background-color:", colorScheme.actionButtonBackgroundPressColor, "!important;}a{color:", theme.colors.actionPrimaryTextDefault, ";}a:focus,a:hover{color:", colorScheme.textHoverColor, ";text-decoration:none;}a:active{color:", colorScheme.textPressColor, " text-decoration:none;}" + (process.env.NODE_ENV === "production" ? "" : ";label:actionButton;"))
  };
};
const levelToIconMap = {
  info: jsx(MegaphoneIcon, {
    "data-testid": "level-info-icon"
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

function _EMOTION_STRINGIFIED_CSS_ERROR__$6() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const chatInputStyles = {
  container: process.env.NODE_ENV === "production" ? {
    name: "ets5eq",
    styles: "background-color:var(--background-primary);padding:var(--spacing-sm);position:relative"
  } : {
    name: "t01lrg-container",
    styles: "background-color:var(--background-primary);padding:var(--spacing-sm);position:relative;label:container;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$6
  },
  suggestionButtons: process.env.NODE_ENV === "production" ? {
    name: "zsd1o9",
    styles: "display:flex;gap:var(--spacing-sm);margin-bottom:var(--spacing-sm)"
  } : {
    name: "3tz5r6-suggestionButtons",
    styles: "display:flex;gap:var(--spacing-sm);margin-bottom:var(--spacing-sm);label:suggestionButtons;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$6
  },
  textArea: process.env.NODE_ENV === "production" ? {
    name: "ge1ym1",
    styles: "min-width:100%;max-height:150px !important"
  } : {
    name: "hcagyb-textArea",
    styles: "min-width:100%;max-height:150px !important;label:textArea;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$6
  },
  button: process.env.NODE_ENV === "production" ? {
    name: "1c60og2",
    styles: "position:absolute;bottom:calc(var(--spacing-sm) + 4px);right:calc(var(--spacing-sm) + 4px);transform:scaleX(-1)"
  } : {
    name: "1h7quuz-button",
    styles: "position:absolute;bottom:calc(var(--spacing-sm) + 4px);right:calc(var(--spacing-sm) + 4px);transform:scaleX(-1);label:button;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$6
  }
};
const ChatInput = _ref => {
  let {
    className,
    onSubmit,
    textAreaProps,
    suggestionButtons
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  const [value, setValue] = React__default.useState('');
  const handleSubmit = () => {
    if (!value) {
      return;
    }
    onSubmit === null || onSubmit === void 0 || onSubmit(value);
    setValue('');
  };
  return jsxs("div", {
    css: chatInputStyles.container,
    style: {
      ['--spacing-sm']: `${theme.spacing.sm}px`,
      ['--background-primary']: theme.colors.backgroundPrimary,
      ['--border-decorative']: theme.colors.borderDecorative
    },
    className: className,
    children: [suggestionButtons && jsx("div", {
      css: chatInputStyles.suggestionButtons,
      children: suggestionButtons
    }), jsx(Input.TextArea, {
      autoSize: true,
      value: value,
      onChange: e => setValue(e.target.value),
      placeholder: "Send a message",
      css: chatInputStyles.textArea,
      onKeyDown: e => {
        if (e.key === 'Enter' && !e.metaKey && !e.ctrlKey && !e.shiftKey) {
          e.preventDefault();
          handleSubmit();
        }
      },
      ...textAreaProps
    }), jsx(Button$1, {
      componentId: "codegen_design-system_src_development_chatui_chatinput.tsx_83",
      size: "small",
      css: chatInputStyles.button,
      icon: jsx(CursorIcon, {}),
      onClick: handleSubmit
    })]
  });
};
var ChatInput$1 = ChatInput;

function _EMOTION_STRINGIFIED_CSS_ERROR__$5() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const snippetStyles = {
  container: process.env.NODE_ENV === "production" ? {
    name: "emyse7",
    styles: "margin:var(--spacing-md) 0"
  } : {
    name: "15luuk2-container",
    styles: "margin:var(--spacing-md) 0;label:container;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$5
  },
  header: process.env.NODE_ENV === "production" ? {
    name: "dtwnbh",
    styles: "background-color:var(--color-grey700);color:var(--color-grey300);border-radius:var(--border-radius) var(--border-radius) 0 0;padding:var(--spacing-sm) var(--spacing-sm) var(--spacing-sm) var(--spacing-md);display:flex;justify-content:space-between;align-items:center"
  } : {
    name: "1o07vd-header",
    styles: "background-color:var(--color-grey700);color:var(--color-grey300);border-radius:var(--border-radius) var(--border-radius) 0 0;padding:var(--spacing-sm) var(--spacing-sm) var(--spacing-sm) var(--spacing-md);display:flex;justify-content:space-between;align-items:center;label:header;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$5
  },
  buttonSvg: process.env.NODE_ENV === "production" ? {
    name: "s8x62f",
    styles: "svg{color:var(--color-grey300);}"
  } : {
    name: "g0ytlm-buttonSvg",
    styles: "svg{color:var(--color-grey300);};label:buttonSvg;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$5
  },
  code: process.env.NODE_ENV === "production" ? {
    name: "u1l3ou",
    styles: "padding:var(--spacing-sm) var(--spacing-md);margin:0;border-radius:0 0 var(--border-radius) var(--border-radius);background-color:var(--color-grey800);color:var(--color-grey300)"
  } : {
    name: "gdk9wh-code",
    styles: "padding:var(--spacing-sm) var(--spacing-md);margin:0;border-radius:0 0 var(--border-radius) var(--border-radius);background-color:var(--color-grey800);color:var(--color-grey300);label:code;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$5
  }
};
const CodeSnippet = _ref => {
  let {
    language,
    buttons,
    children
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  return jsxs("div", {
    css: snippetStyles.container,
    style: {
      ['--spacing-sm']: `${theme.spacing.sm}px`,
      ['--spacing-md']: `${theme.spacing.md}px`,
      ['--border-radius']: `${theme.general.borderRadiusBase}px`,
      ['--color-grey700']: theme.colors.grey700,
      ['--color-grey300']: theme.colors.grey300,
      ['--color-grey800']: theme.colors.grey800
    },
    children: [jsxs("div", {
      css: snippetStyles.header,
      children: [jsx("span", {
        children: language
      }), jsx("div", {
        css: snippetStyles.buttonSvg,
        children: buttons
      })]
    }), jsx("pre", {
      css: snippetStyles.code,
      children: children
    })]
  });
};
var CodeSnippet$1 = CodeSnippet;

function _EMOTION_STRINGIFIED_CSS_ERROR__$4() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const feedbackStyles = {
  container: process.env.NODE_ENV === "production" ? {
    name: "1laz9o6",
    styles: "display:flex;justify-content:space-between;align-items:flex-end"
  } : {
    name: "1dbf7uj-container",
    styles: "display:flex;justify-content:space-between;align-items:flex-end;label:container;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$4
  },
  feedbackPrompt: process.env.NODE_ENV === "production" ? {
    name: "1hsyf68",
    styles: "display:flex;flex-direction:column;gap:var(--spacing-sm)"
  } : {
    name: "1yi757f-feedbackPrompt",
    styles: "display:flex;flex-direction:column;gap:var(--spacing-sm);label:feedbackPrompt;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$4
  },
  feedbackOptions: process.env.NODE_ENV === "production" ? {
    name: "1s3radb",
    styles: "display:flex;gap:var(--spacing-sm)"
  } : {
    name: "125q18r-feedbackOptions",
    styles: "display:flex;gap:var(--spacing-sm);label:feedbackOptions;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$4
  }
};
const Feedback = _ref => {
  let {
    onFeedback
  } = _ref;
  const [isVisible, setIsVisible] = useState(true);
  const {
    theme
  } = useDesignSystemTheme();
  if (!isVisible) {
    return null;
  }
  return jsxs("div", {
    css: feedbackStyles.container,
    style: {
      ['--spacing-sm']: `${theme.spacing.sm}px`,
      ['--text-secondary']: theme.colors.textSecondary
    },
    children: [jsxs("div", {
      css: feedbackStyles.feedbackPrompt,
      children: [jsx(Typography.Text, {
        style: {
          color: theme.colors.textSecondary
        },
        children: "Was this response:"
      }), jsxs("div", {
        css: feedbackStyles.feedbackOptions,
        children: [jsx(Button$1, {
          componentId: "codegen_design-system_src_development_chatui_feedback.tsx_57",
          icon: jsx(FaceSmileIcon, {}),
          onClick: () => {
            onFeedback('Better');
            setIsVisible(false);
          },
          children: "Better"
        }), jsx(Button$1, {
          componentId: "codegen_design-system_src_development_chatui_feedback.tsx_66",
          icon: jsx(FaceNeutralIcon, {}),
          onClick: () => {
            onFeedback('Same');
            setIsVisible(false);
          },
          children: "Same"
        }), jsx(Button$1, {
          componentId: "codegen_design-system_src_development_chatui_feedback.tsx_75",
          icon: jsx(FaceFrownIcon, {}),
          onClick: () => {
            onFeedback('Worse');
            setIsVisible(false);
          },
          children: "Worse"
        })]
      })]
    }), jsx(Button$1, {
      componentId: "codegen_design-system_src_development_chatui_feedback.tsx_86",
      icon: jsx(CloseIcon, {}),
      onClick: () => setIsVisible(false)
    })]
  });
};
var Feedback$1 = Feedback;

const messageStyles = {
  container: /*#__PURE__*/css({
    display: 'flex',
    flexDirection: 'column',
    gap: 'var(--spacing-sm)',
    paddingTop: 'var(--spacing-md)',
    paddingRight: 'var(--spacing-sm)',
    paddingBottom: 'var(--spacing-md)',
    paddingLeft: 'calc(var(--spacing-md) * 2)',
    backgroundColor: 'var(--background)',
    borderBottom: `1px solid var(--border-decorative)`
  }, process.env.NODE_ENV === "production" ? "" : ";label:container;")
};
const Message = _ref => {
  let {
    isActiveUser,
    children
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx("div", {
    css: messageStyles.container,
    style: {
      ['--spacing-sm']: `${theme.spacing.sm}px`,
      ['--spacing-md']: `${theme.spacing.md}px`,
      ['--background']: isActiveUser ? theme.colors.backgroundPrimary : theme.colors.backgroundSecondary,
      ['--border-decorative']: theme.colors.borderDecorative
    },
    children: children
  });
};
var Message$1 = Message;

function _EMOTION_STRINGIFIED_CSS_ERROR__$3() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const buttonStyles = {
  container: process.env.NODE_ENV === "production" ? {
    name: "f9548c",
    styles: "position:relative;display:inline-block;width:max-content"
  } : {
    name: "12rixe4-container",
    styles: "position:relative;display:inline-block;width:max-content;label:container;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$3
  },
  checkIcon: process.env.NODE_ENV === "production" ? {
    name: "441ysl",
    styles: "position:absolute;top:50%;left:50%;transform:translate(-50%, -50%);pointer-events:none;svg{color:var(--checkmark-color);}"
  } : {
    name: "hmgfz4-checkIcon",
    styles: "position:absolute;top:50%;left:50%;transform:translate(-50%, -50%);pointer-events:none;svg{color:var(--checkmark-color);};label:checkIcon;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$3
  }
};
const MessageActionButton = props => {
  const [showCheck, setShowCheck] = useState(false);
  const {
    theme
  } = useDesignSystemTheme();
  const handleClick = e => {
    var _props$onClick;
    setShowCheck(true);
    (_props$onClick = props.onClick) === null || _props$onClick === void 0 || _props$onClick.call(props, e);
  };
  if (props.children) {
    throw new Error('MessageActionButton: Children not supported; use as icon-only button.');
  }
  useEffect(() => {
    let timeoutId;
    if (showCheck) {
      timeoutId = setTimeout(() => {
        setShowCheck(false);
      }, 750);
    }
    return () => {
      clearTimeout(timeoutId);
    };
  }, [showCheck]);
  return jsxs("span", {
    css: buttonStyles.container,
    style: {
      ['--checkmark-color']: theme.colors.green500
    },
    children: [jsx(Button$1, {
      size: "small",
      ...props,
      onClick: handleClick,
      css: /*#__PURE__*/css({
        svg: {
          transition: showCheck ? 'none' : 'opacity 350ms',
          opacity: showCheck ? '0' : '1'
        }
      }, process.env.NODE_ENV === "production" ? "" : ";label:MessageActionButton;")
    }), jsx(CheckIcon, {
      css: buttonStyles.checkIcon,
      style: {
        transition: showCheck ? 'none' : 'opacity 350ms',
        opacity: showCheck ? '1' : '0'
      }
    })]
  });
};
var MessageActionButton$1 = MessageActionButton;

const MessageBody = _ref => {
  let {
    children
  } = _ref;
  const {
    getPrefixedClassName
  } = useDesignSystemTheme();
  const typographyClassname = getPrefixedClassName('typography');
  return jsx("div", {
    css: /*#__PURE__*/css({
      [`& .${typographyClassname}:last-of-type`]: {
        marginBottom: 0
      }
    }, process.env.NODE_ENV === "production" ? "" : ";label:MessageBody;"),
    children: children
  });
};
var MessageBody$1 = MessageBody;

function _EMOTION_STRINGIFIED_CSS_ERROR__$2() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const headerStyles = {
  container: process.env.NODE_ENV === "production" ? {
    name: "30lgt6",
    styles: "display:flex;justify-content:space-between;align-items:center;position:relative"
  } : {
    name: "10tss9b-container",
    styles: "display:flex;justify-content:space-between;align-items:center;position:relative;label:container;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$2
  },
  leftContent: process.env.NODE_ENV === "production" ? {
    name: "sz7nmf",
    styles: "display:flex;align-items:center;gap:var(--spacing-sm)"
  } : {
    name: "11wiry4-leftContent",
    styles: "display:flex;align-items:center;gap:var(--spacing-sm);label:leftContent;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$2
  },
  rightContent: process.env.NODE_ENV === "production" ? {
    name: "s5xdrg",
    styles: "display:flex;align-items:center"
  } : {
    name: "6oh64j-rightContent",
    styles: "display:flex;align-items:center;label:rightContent;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$2
  },
  avatar: /*#__PURE__*/css({
    position: 'absolute',
    left: -22,
    top: 2,
    backgroundSize: 'cover',
    backgroundPosition: 'center',
    width: 'var(--spacing-md)',
    height: 'var(--spacing-md)',
    borderRadius: '50%'
  }, process.env.NODE_ENV === "production" ? "" : ";label:avatar;")
};
const avatarDefaultBackgroundImage = 'linear-gradient(180deg, #FFD983 0%, #FFB800 100%)';
const MessageHeader = _ref => {
  let {
    userName,
    customHeaderIcon,
    avatarURL,
    leftContent,
    rightContent
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  return jsxs("div", {
    css: headerStyles.container,
    style: {
      ['--spacing-sm']: `${theme.spacing.sm}px`,
      ['--spacing-md']: `${theme.spacing.md}px`
    },
    children: [jsxs("div", {
      css: headerStyles.leftContent,
      children: [customHeaderIcon ? customHeaderIcon : jsx("div", {
        css: headerStyles.avatar,
        style: {
          backgroundImage: avatarURL ? `url(${avatarURL}), ${avatarDefaultBackgroundImage}` : avatarDefaultBackgroundImage
        }
      }), jsx(Typography.Text, {
        bold: true,
        children: userName
      }), leftContent]
    }), jsx("div", {
      css: headerStyles.rightContent,
      children: rightContent
    })]
  });
};
var MessageHeader$1 = MessageHeader;

function _EMOTION_STRINGIFIED_CSS_ERROR__$1() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const paginationStyles = {
  container: process.env.NODE_ENV === "production" ? {
    name: "1lyitb8",
    styles: "display:flex;align-items:center;justify-items:center;gap:var(--spacing-sm)"
  } : {
    name: "tx606a-container",
    styles: "display:flex;align-items:center;justify-items:center;gap:var(--spacing-sm);label:container;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$1
  },
  button: process.env.NODE_ENV === "production" ? {
    name: "waw0s4",
    styles: "border:none;background-color:transparent;padding:0;display:flex;height:var(--spacing-md);align-items:center"
  } : {
    name: "18u25xw-button",
    styles: "border:none;background-color:transparent;padding:0;display:flex;height:var(--spacing-md);align-items:center;label:button;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$1
  }
};
const MessagePagination = _ref => {
  let {
    onPrevious,
    onNext,
    current,
    total
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  return jsxs("div", {
    css: paginationStyles.container,
    style: {
      ['--spacing-sm']: `${theme.spacing.sm}px`,
      ['--spacing-md']: `${theme.spacing.md}px`
    },
    children: [jsx("button", {
      css: paginationStyles.button,
      onClick: onPrevious,
      style: {
        color: current === 1 ? theme.colors.actionDisabledText : theme.colors.textSecondary,
        cursor: current === 1 ? 'arrow' : 'pointer'
      },
      children: jsx(ChevronLeftIcon, {})
    }), jsx(Typography.Text, {
      style: {
        color: theme.colors.textSecondary
      },
      children: `${current}/${total}`
    }), jsx("button", {
      css: paginationStyles.button,
      onClick: onNext,
      style: {
        color: current === total ? theme.colors.actionDisabledText : theme.colors.textSecondary,
        cursor: current === total ? 'arrow' : 'pointer'
      },
      children: jsx(ChevronRightIcon, {})
    })]
  });
};
var MessagePagination$1 = MessagePagination;

const ChatUI = {
  MessageActionButton: MessageActionButton$1,
  MessageHeader: MessageHeader$1,
  MessageBody: MessageBody$1,
  MessagePagination: MessagePagination$1,
  Message: Message$1,
  Feedback: Feedback$1,
  CodeSnippet: CodeSnippet$1,
  ChatInput: ChatInput$1
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
const Root$1 = /*#__PURE__*/forwardRef((props, ref) => {
  return jsx(RadixSlider.Root, {
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
    borderRadius: theme.borders.borderRadiusMd,
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

export { BANNER_MAX_HEIGHT, BANNER_MIN_HEIGHT, Banner, ChatUI, Slider, Toolbar };
//# sourceMappingURL=development.js.map
