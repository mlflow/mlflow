import { css } from '@emotion/react';
import React__default, { useState, useEffect, forwardRef } from 'react';
import { u as useDesignSystemTheme, e as Input, B as Button, N as CursorIcon, j as Typography, Q as FaceSmileIcon, P as FaceNeutralIcon, O as FaceFrownIcon, C as CloseIcon, f as CheckIcon, o as ChevronLeftIcon, a as ChevronRightIcon, $ as dropdownItemStyles, c as useDesignSystemContext, l as dropdownContentStyles, a0 as dropdownSeparatorStyles } from './DropdownMenu-d39d9283.js';
import { jsxs, jsx } from '@emotion/react/jsx-runtime';
import * as RadixSlider from '@radix-ui/react-slider';
import { ContextMenu as ContextMenu$1, ContextMenuTrigger, ContextMenuItemIndicator, ContextMenuGroup, ContextMenuRadioGroup, ContextMenuArrow, ContextMenuSub, ContextMenuSubTrigger, ContextMenuPortal, ContextMenuContent, ContextMenuSubContent, ContextMenuItem, ContextMenuCheckboxItem, ContextMenuRadioItem, ContextMenuLabel, ContextMenuSeparator } from '@radix-ui/react-context-menu';
import 'antd';
import '@radix-ui/react-dropdown-menu';
import 'chroma-js';
import 'lodash/isNil';
import 'lodash/endsWith';
import 'lodash/isBoolean';
import 'lodash/isNumber';
import 'lodash/isString';
import 'lodash/mapValues';
import '@emotion/unitless';
import '@ant-design/icons';
import 'lodash/uniqueId';

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
    onSubmit === null || onSubmit === void 0 ? void 0 : onSubmit(value);
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
    }), jsx(Button, {
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
        children: [jsx(Button, {
          icon: jsx(FaceSmileIcon, {}),
          onClick: () => {
            onFeedback('Better');
            setIsVisible(false);
          },
          children: "Better"
        }), jsx(Button, {
          icon: jsx(FaceNeutralIcon, {}),
          onClick: () => {
            onFeedback('Same');
            setIsVisible(false);
          },
          children: "Same"
        }), jsx(Button, {
          icon: jsx(FaceFrownIcon, {}),
          onClick: () => {
            onFeedback('Worse');
            setIsVisible(false);
          },
          children: "Worse"
        })]
      })]
    }), jsx(Button, {
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
    (_props$onClick = props.onClick) === null || _props$onClick === void 0 ? void 0 : _props$onClick.call(props, e);
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
    children: [jsx(Button, {
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
      children: [jsx("div", {
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
const getRootStyles = () => {
  return _ref;
};
const Root$1 = /*#__PURE__*/forwardRef((props, ref) => {
  return jsx(RadixSlider.Root, {
    css: getRootStyles(),
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

const Root = ContextMenu$1;
const Trigger = ContextMenuTrigger;
const ItemIndicator = ContextMenuItemIndicator;
const Group = ContextMenuGroup;
const RadioGroup = ContextMenuRadioGroup;
const Arrow = ContextMenuArrow;
const Sub = ContextMenuSub;
const SubTrigger = _ref => {
  let {
    children,
    ...props
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(ContextMenuSubTrigger, {
    ...props,
    css: dropdownItemStyles(theme),
    children: children
  });
};
const Content = _ref2 => {
  let {
    children,
    minWidth,
    ...childrenProps
  } = _ref2;
  const {
    getPopupContainer
  } = useDesignSystemContext();
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(ContextMenuPortal, {
    container: getPopupContainer && getPopupContainer(),
    children: jsx(ContextMenuContent, {
      ...childrenProps,
      css: [dropdownContentStyles(theme), {
        minWidth
      }, process.env.NODE_ENV === "production" ? "" : ";label:Content;"],
      children: children
    })
  });
};
const SubContent = _ref3 => {
  let {
    children,
    minWidth,
    ...childrenProps
  } = _ref3;
  const {
    getPopupContainer
  } = useDesignSystemContext();
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(ContextMenuPortal, {
    container: getPopupContainer && getPopupContainer(),
    children: jsx(ContextMenuSubContent, {
      ...childrenProps,
      css: [dropdownContentStyles(theme), {
        minWidth
      }, process.env.NODE_ENV === "production" ? "" : ";label:SubContent;"],
      children: children
    })
  });
};
const Item = _ref4 => {
  let {
    children,
    ...props
  } = _ref4;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(ContextMenuItem, {
    ...props,
    css: dropdownItemStyles(theme),
    children: children
  });
};
const CheckboxItem = _ref5 => {
  let {
    children,
    ...props
  } = _ref5;
  const {
    theme
  } = useDesignSystemTheme();
  return jsxs(ContextMenuCheckboxItem, {
    ...props,
    css: dropdownItemStyles(theme),
    children: [jsx(ContextMenuItemIndicator, {
      css: itemIndicatorStyles(theme),
      children: jsx(CheckIcon, {})
    }), !props.checked && jsx("div", {
      style: {
        width: theme.general.iconFontSize + theme.spacing.xs
      }
    }), children]
  });
};
const RadioItem = _ref6 => {
  let {
    children,
    ...props
  } = _ref6;
  const {
    theme
  } = useDesignSystemTheme();
  return jsxs(ContextMenuRadioItem, {
    ...props,
    css: [dropdownItemStyles(theme), {
      '&[data-state="unchecked"]': {
        paddingLeft: theme.general.iconFontSize + theme.spacing.xs + theme.spacing.sm
      }
    }, process.env.NODE_ENV === "production" ? "" : ";label:RadioItem;"],
    children: [jsx(ContextMenuItemIndicator, {
      css: itemIndicatorStyles(theme),
      children: jsx(CheckIcon, {})
    }), children]
  });
};
const Label = _ref7 => {
  let {
    children,
    ...props
  } = _ref7;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(ContextMenuLabel, {
    ...props,
    css: /*#__PURE__*/css({
      color: theme.colors.textSecondary,
      padding: `${theme.spacing.sm - 2}px ${theme.spacing.sm}px`
    }, process.env.NODE_ENV === "production" ? "" : ";label:Label;"),
    children: children
  });
};
const Hint = _ref8 => {
  let {
    children
  } = _ref8;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx("span", {
    css: /*#__PURE__*/css({
      display: 'inline-flex',
      marginLeft: 'auto',
      paddingLeft: theme.spacing.sm
    }, process.env.NODE_ENV === "production" ? "" : ";label:Hint;"),
    children: children
  });
};
const Separator = () => {
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(ContextMenuSeparator, {
    css: dropdownSeparatorStyles(theme)
  });
};
const itemIndicatorStyles = theme => /*#__PURE__*/css({
  display: 'inline-flex',
  paddingRight: theme.spacing.xs
}, process.env.NODE_ENV === "production" ? "" : ";label:itemIndicatorStyles;");
const ContextMenu = {
  Root,
  Trigger,
  Label,
  Item,
  Group,
  RadioGroup,
  CheckboxItem,
  RadioItem,
  Arrow,
  Separator,
  Sub,
  SubTrigger,
  SubContent,
  Content,
  Hint
};

export { Arrow, ChatUI, CheckboxItem, Content, ContextMenu, Group, Hint, Item, ItemIndicator, Label, RadioGroup, RadioItem, Root, Separator, Slider, Sub, SubContent, SubTrigger, Trigger, itemIndicatorStyles };
//# sourceMappingURL=development.js.map
