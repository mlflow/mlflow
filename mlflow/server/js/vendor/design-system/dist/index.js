import * as React from 'react';
import React__default, { createContext, useContext, useMemo, useEffect, useRef, forwardRef, useState, useImperativeHandle, Children, useCallback } from 'react';
import AntDIcon from '@ant-design/icons';
import { ThemeProvider, css, keyframes, ClassNames } from '@emotion/react';
import { jsx, jsxs, Fragment } from '@emotion/react/jsx-runtime';
import { g as getTheme, a as getClassNamePrefix, b as getPrefixedClassNameFromTheme, u as useDesignSystemTheme, l as lightColorList } from './useDesignSystemTheme-c867a35d.js';
export { W as WithDesignSystemThemeHoc, u as useDesignSystemTheme } from './useDesignSystemTheme-c867a35d.js';
import { notification, ConfigProvider, Collapse, Alert as Alert$1, AutoComplete as AutoComplete$1, Breadcrumb as Breadcrumb$1, Button as Button$1, Checkbox as Checkbox$1, DatePicker, Tooltip as Tooltip$1, Input as Input$1, Typography as Typography$1, Dropdown as Dropdown$1, Form as Form$1, Radio as Radio$1, Select as Select$1, Col as Col$1, Row as Row$1, Space as Space$1, Layout as Layout$1, Pagination as Pagination$1, Table as Table$1, Menu as Menu$1, Modal as Modal$1, Popover as Popover$2, Skeleton as Skeleton$1, Switch as Switch$1, Tabs as Tabs$1, Tree as Tree$1 } from 'antd';
import classnames from 'classnames';
import _isNil from 'lodash/isNil';
import _endsWith from 'lodash/endsWith';
import _isBoolean from 'lodash/isBoolean';
import _isNumber from 'lodash/isNumber';
import _isString from 'lodash/isString';
import _mapValues from 'lodash/mapValues';
import unitless from '@emotion/unitless';
import * as Popover$1 from '@radix-ui/react-popover';
import * as DialogPrimitive from '@radix-ui/react-dialog';
import * as DropdownMenu$1 from '@radix-ui/react-dropdown-menu';
import { useController } from 'react-hook-form';
import * as Toast from '@radix-ui/react-toast';
import { ResizableBox } from 'react-resizable';
import * as Toggle from '@radix-ui/react-toggle';
import 'chroma-js';

const DuboisContextDefaults = {
  enableAnimation: false,
  // Prefer to use useDesignSystemTheme.getPrefixedClassName instead
  getPrefixCls: suffix => suffix ? `du-bois-${suffix}` : 'du-bois',
  flags: {}
};
const DesignSystemThemeContext = /*#__PURE__*/createContext({
  isDarkMode: false
});
const DesignSystemContext = /*#__PURE__*/createContext(DuboisContextDefaults);
const DU_BOIS_ENABLE_ANIMATION_CLASSNAME = 'du-bois-enable-animation';
function getAnimationCss(enableAnimation) {
  const disableAnimationCss = {
    animationDuration: '0s !important',
    transition: 'none !important'
  };
  return enableAnimation ? {} : {
    // Apply to the current element
    ...disableAnimationCss,
    '&::before': disableAnimationCss,
    '&::after': disableAnimationCss,
    // Also apply to all child elements with a class that starts with our prefix
    [`[class*=du-bois]:not(.${DU_BOIS_ENABLE_ANIMATION_CLASSNAME}, .${DU_BOIS_ENABLE_ANIMATION_CLASSNAME} *)`]: {
      ...disableAnimationCss,
      // Also target any pseudo-elements associated with those elements, since these can also be animated.
      '&::before': disableAnimationCss,
      '&::after': disableAnimationCss
    }
  };
}
const DesignSystemProviderPropsContext = /*#__PURE__*/React__default.createContext(null);
const AntDConfigProviderPropsContext = /*#__PURE__*/React__default.createContext(null);

/** Only to be accessed by SupportsDuBoisThemes, except for special exceptions like tests and storybook. Ask in #dubois first if you need to use it. */
const DesignSystemThemeProvider = _ref => {
  let {
    isDarkMode = false,
    children
  } = _ref;
  return jsx(DesignSystemThemeContext.Provider, {
    value: {
      isDarkMode
    },
    children: children
  });
};
const DesignSystemProvider = _ref2 => {
  let {
    isDarkMode: deprecatedDarkModeProp,
    children,
    enableAnimation = false,
    zIndexBase = 1000,
    getPopupContainer,
    flags = {
      USE_NEW_ICONS: true
    }
  } = _ref2;
  const {
    isDarkMode: contextDarkModeVal
  } = useContext(DesignSystemThemeContext);
  const isDarkMode = deprecatedDarkModeProp !== null && deprecatedDarkModeProp !== void 0 ? deprecatedDarkModeProp : contextDarkModeVal;
  const theme = useMemo(() => getTheme(isDarkMode, {
    enableAnimation,
    zIndexBase
  }),
  // TODO: revisit this
  // eslint-disable-next-line react-hooks/exhaustive-deps
  [isDarkMode, zIndexBase]);
  const providerPropsContext = useMemo(() => ({
    isDarkMode,
    enableAnimation,
    zIndexBase,
    getPopupContainer,
    flags
  }), [isDarkMode, enableAnimation, zIndexBase, getPopupContainer, flags]);
  const classNamePrefix = getClassNamePrefix(theme);
  const value = useMemo(() => {
    return {
      enableAnimation,
      isDarkMode,
      getPrefixCls: suffix => getPrefixedClassNameFromTheme(theme, suffix),
      getPopupContainer,
      flags
    };
  }, [enableAnimation, theme, isDarkMode, getPopupContainer, flags]);
  useEffect(() => {
    return () => {
      // when the design system context is unmounted, make sure the notification cache is also cleaned up
      notification.destroy();
    };
  }, []);
  return jsx(DesignSystemProviderPropsContext.Provider, {
    value: providerPropsContext,
    children: jsx(ThemeProvider, {
      theme: theme,
      children: jsx(AntDConfigProviderPropsContext.Provider, {
        value: {
          prefixCls: classNamePrefix,
          getPopupContainer
        },
        children: jsx(DesignSystemContext.Provider, {
          value: value,
          children: children
        })
      })
    })
  });
};
const ApplyDesignSystemContextOverrides = _ref3 => {
  let {
    enableAnimation,
    zIndexBase,
    getPopupContainer,
    flags,
    children
  } = _ref3;
  const parentDesignSystemProviderProps = useContext(DesignSystemProviderPropsContext);
  if (parentDesignSystemProviderProps === null) {
    throw new Error(`ApplyDesignSystemContextOverrides cannot be used standalone - DesignSystemProvider must exist in the React context`);
  }
  const newProps = useMemo(() => ({
    ...parentDesignSystemProviderProps,
    enableAnimation: enableAnimation !== null && enableAnimation !== void 0 ? enableAnimation : parentDesignSystemProviderProps.enableAnimation,
    zIndexBase: zIndexBase !== null && zIndexBase !== void 0 ? zIndexBase : parentDesignSystemProviderProps.zIndexBase,
    getPopupContainer: getPopupContainer !== null && getPopupContainer !== void 0 ? getPopupContainer : parentDesignSystemProviderProps.getPopupContainer,
    flags: {
      ...parentDesignSystemProviderProps.flags,
      ...flags
    }
  }), [parentDesignSystemProviderProps, enableAnimation, zIndexBase, getPopupContainer, flags]);
  return jsx(DesignSystemProvider, {
    ...newProps,
    children: children
  });
};

// This is a more-specific version of `ApplyDesignSystemContextOverrides` that only allows overriding the flags.
const ApplyDesignSystemFlags = _ref4 => {
  let {
    flags,
    children
  } = _ref4;
  const parentDesignSystemProviderProps = useContext(DesignSystemProviderPropsContext);
  if (parentDesignSystemProviderProps === null) {
    throw new Error(`ApplyDesignSystemFlags cannot be used standalone - DesignSystemProvider must exist in the React context`);
  }
  const newProps = useMemo(() => ({
    ...parentDesignSystemProviderProps,
    flags: {
      ...parentDesignSystemProviderProps.flags,
      ...flags
    }
  }), [parentDesignSystemProviderProps, flags]);
  return jsx(DesignSystemProvider, {
    ...newProps,
    children: children
  });
};
const DesignSystemAntDConfigProvider = _ref5 => {
  let {
    children
  } = _ref5;
  const antdContext = useAntDConfigProviderContext();
  return jsx(ConfigProvider, {
    ...antdContext,
    children: children
  });
};
const useAntDConfigProviderContext = () => {
  var _useContext;
  return (_useContext = useContext(AntDConfigProviderPropsContext)) !== null && _useContext !== void 0 ? _useContext : {
    prefixCls: undefined
  };
};

/**
 * When using AntD components inside Design System wrapper components (e.g. Modal, Collapse etc),
 * we don't want Design System's prefix class to override them.
 *
 * Since all Design System's components have are wrapped with DesignSystemAntDConfigProvider,
 * this won't affect their own prefixCls, but just allow nested antd components to keep their ant prefix.
 */
const RestoreAntDDefaultClsPrefix = _ref6 => {
  let {
    children
  } = _ref6;
  return jsx(ConfigProvider, {
    prefixCls: "ant",
    children: children
  });
};

function useDesignSystemContext() {
  return useContext(DesignSystemContext);
}

function useDesignSystemFlags() {
  const context = useDesignSystemContext();
  return context.flags;
}

/**
 * A helper hook that allows quick creation of theme-dependent styles. 
 * Results in more compact code than using useMemo and 
 * useDesignSystemTheme separately.
 * 
 * @example 
 * const styles = useThemedStyles((theme) => ({
 *   overlay: {
 *     backgroundColor: theme.colors.backgroundPrimary,
 *     borderRadius: theme.borders.borderRadiusMd,
 *   },
 *   wrapper: {
 *     display: 'flex',
 *     gap: theme.spacing.md,
 *   },
 * }));

 * <div css={styles.overlay}>...</div>
 * 
 * @param styleFactory Factory function that accepts theme object as a parameter and returns
 *     the style object. **Note**: factory function body is being memoized internally and is intended
 *     to be used only for simple style objects that depend solely on the theme. If you want to use
 *     styles that change depending on external values (state, props etc.) you should use
 *     `useDesignSystemTheme` directly with  your own reaction mechanism.
 * @returns The constructed style object
 */
const useThemedStyles = styleFactory => {
  const {
    theme
  } = useDesignSystemTheme();

  // We can assume that the factory function won't change and we're
  // observing theme changes only.
  const styleFactoryRef = useRef(styleFactory);
  return useMemo(() => styleFactoryRef.current(theme), [theme]);
};

const getIconEmotionStyles = (theme, useNewIcons) => {
  return /*#__PURE__*/css({
    ...(useNewIcons && {
      fontSize: theme.general.iconFontSizeNew
    })
  }, process.env.NODE_ENV === "production" ? "" : ";label:getIconEmotionStyles;");
};
const getIconValidationColor = (theme, color) => {
  switch (color) {
    case 'success':
      return theme.colors.textValidationSuccess;
    case 'warning':
      return theme.colors.textValidationWarning;
    case 'danger':
      return theme.colors.textValidationDanger;
    default:
      return color;
  }
};
const Icon = props => {
  const {
    component: Component,
    dangerouslySetAntdProps,
    color,
    style,
    ...otherProps
  } = props;
  const {
    theme
  } = useDesignSystemTheme();
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const MemoizedComponent = useMemo(() => Component ? _ref => {
    let {
      fill,
      ...iconProps
    } = _ref;
    return (
      // We don't rely on top-level fills for our colors. Fills are specified
      // with "currentColor" on children of the top-most svg.
      jsx(Component, {
        fill: "none",
        ...iconProps
      })
    );
  } : undefined, [Component]);
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(AntDIcon, {
      css: getIconEmotionStyles(theme, USE_NEW_ICONS),
      component: MemoizedComponent,
      style: {
        color: getIconValidationColor(theme, color),
        ...style
      },
      ...otherProps,
      ...dangerouslySetAntdProps
    })
  });
};

function SvgAlignCenterIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M4 5c-.55 0-1-.45-1-1s.45-1 1-1h16c.55 0 1 .45 1 1s-.45 1-1 1H4zm3 3c0 .55.45 1 1 1h8c.55 0 1-.45 1-1s-.45-1-1-1H8c-.55 0-1 .45-1 1zm13 5H4c-.55 0-1-.45-1-1s.45-1 1-1h16c.55 0 1 .45 1 1s-.45 1-1 1zM7 16c0 .55.45 1 1 1h8c.55 0 1-.45 1-1s-.45-1-1-1H8c-.55 0-1 .45-1 1zm-3 5h16c.55 0 1-.45 1-1s-.45-1-1-1H4c-.55 0-1 .45-1 1s.45 1 1 1z",
      fill: "currentColor"
    })
  });
}
function SvgAlignCenterIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M1 2.5h14V1H1v1.5zM11.5 5.75h-7v-1.5h7v1.5zM15 8.75H1v-1.5h14v1.5zM15 15H1v-1.5h14V15zM4.5 11.75h7v-1.5h-7v1.5z",
      fill: "currentColor"
    })
  });
}
function AlignCenterIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgAlignCenterIconV2 : SvgAlignCenterIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgAlignLeftIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M4 5c-.55 0-1-.45-1-1s.45-1 1-1h16c.55 0 1 .45 1 1s-.45 1-1 1H4zm10 2H4c-.55 0-1 .45-1 1s.45 1 1 1h10c.55 0 1-.45 1-1s-.45-1-1-1zm0 8H4c-.55 0-1 .45-1 1s.45 1 1 1h10c.55 0 1-.45 1-1s-.45-1-1-1zm6-2H4c-.55 0-1-.45-1-1s.45-1 1-1h16c.55 0 1 .45 1 1s-.45 1-1 1zM4 21h16c.55 0 1-.45 1-1s-.45-1-1-1H4c-.55 0-1 .45-1 1s.45 1 1 1z",
      fill: "currentColor"
    })
  });
}
function SvgAlignLeftIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M1 2.5h14V1H1v1.5zM8 5.75H1v-1.5h7v1.5zM1 8.75v-1.5h14v1.5H1zM1 15v-1.5h14V15H1zM1 11.75h7v-1.5H1v1.5z",
      fill: "currentColor"
    })
  });
}
function AlignLeftIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgAlignLeftIconV2 : SvgAlignLeftIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgAlignRightIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M4 5c-.55 0-1-.45-1-1s.45-1 1-1h16c.55 0 1 .45 1 1s-.45 1-1 1H4zm6 4h10c.55 0 1-.45 1-1s-.45-1-1-1H10c-.55 0-1 .45-1 1s.45 1 1 1zm10 4H4c-.55 0-1-.45-1-1s.45-1 1-1h16c.55 0 1 .45 1 1s-.45 1-1 1zm-10 4h10c.55 0 1-.45 1-1s-.45-1-1-1H10c-.55 0-1 .45-1 1s.45 1 1 1zm-6 4h16c.55 0 1-.45 1-1s-.45-1-1-1H4c-.55 0-1 .45-1 1s.45 1 1 1z",
      fill: "currentColor"
    })
  });
}
function SvgAlignRightIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M1 2.5h14V1H1v1.5zM15 5.75H8v-1.5h7v1.5zM1 8.75v-1.5h14v1.5H1zM1 15v-1.5h14V15H1zM8 11.75h7v-1.5H8v1.5z",
      fill: "currentColor"
    })
  });
}
function AlignRightIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgAlignRightIconV2 : SvgAlignRightIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgArrowDownIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M11.005 5.209v11.17l-4.88-4.88c-.39-.39-1.03-.39-1.42 0a.996.996 0 000 1.41l6.59 6.59c.39.39 1.02.39 1.41 0l6.59-6.59a.996.996 0 10-1.41-1.41l-4.88 4.88V5.209c0-.55-.45-1-1-1s-1 .45-1 1z",
      fill: "currentColor"
    })
  });
}
function SvgArrowDownIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M8.03 15.06L1 8.03l1.06-1.06 5.22 5.22V1h1.5v11.19L14 6.97l1.06 1.06-7.03 7.03z",
      fill: "currentColor"
    })
  });
}
function ArrowDownIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgArrowDownIconV2 : SvgArrowDownIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgArrowLeftIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M18.791 11.005H7.621l4.88-4.88c.39-.39.39-1.03 0-1.42a.996.996 0 00-1.41 0l-6.59 6.59a.996.996 0 000 1.41l6.59 6.59a.996.996 0 101.41-1.41l-4.88-4.88h11.17c.55 0 1-.45 1-1s-.45-1-1-1z",
      fill: "currentColor"
    })
  });
}
function SvgArrowLeftIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M1 8.03L8.03 1l1.061 1.06-5.22 5.22h11.19v1.5H3.87L9.091 14l-1.06 1.06L1 8.03z",
      fill: "currentColor"
    })
  });
}
function ArrowLeftIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgArrowLeftIconV2 : SvgArrowLeftIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgArrowRightIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M5.209 13h11.17l-4.88 4.88c-.39.39-.39 1.03 0 1.42.39.39 1.02.39 1.41 0l6.59-6.59a.996.996 0 000-1.41l-6.58-6.6a.996.996 0 00-1.41 0 .996.996 0 000 1.41l4.87 4.89H5.209c-.55 0-1 .45-1 1s.45 1 1 1z",
      fill: "currentColor"
    })
  });
}
function SvgArrowRightIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M15.06 8.03l-7.03 7.03L6.97 14l5.22-5.22H1v-1.5h11.19L6.97 2.06 8.03 1l7.03 7.03z",
      fill: "currentColor"
    })
  });
}
function ArrowRightIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgArrowRightIconV2 : SvgArrowRightIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgArrowUpIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M13 18.791V7.621l4.88 4.88c.39.39 1.03.39 1.42 0a.996.996 0 000-1.41l-6.59-6.59a.996.996 0 00-1.41 0l-6.6 6.58a.996.996 0 101.41 1.41L11 7.622v11.17c0 .55.45 1 1 1s1-.45 1-1z",
      fill: "currentColor"
    })
  });
}
function SvgArrowUpIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M8.03 1l7.03 7.03L14 9.091l-5.22-5.22v11.19h-1.5V3.87l-5.22 5.22L1 8.031 8.03 1z",
      fill: "currentColor"
    })
  });
}
function ArrowUpIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgArrowUpIconV2 : SvgArrowUpIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgArrowsUpDownIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M5.856 6.145l2.79-2.79c.19-.19.51-.19.7 0l2.79 2.79c.32.31.1.85-.35.85h-1.79v6.01c0 .55-.45 1-1 1s-1-.45-1-1v-6.01h-1.79c-.45 0-.67-.54-.35-.85zm10.14 4.86v6.01h1.8c.44 0 .67.54.35.85l-2.79 2.78c-.2.19-.51.19-.71 0l-2.79-2.78c-.32-.31-.1-.85.35-.85h1.79v-6.01c0-.55.45-1 1-1s1 .45 1 1z",
      fill: "currentColor"
    })
  });
}
function SvgArrowsUpDownIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M5.03 1L1 5.03l1.06 1.061 2.22-2.22v6.19h1.5V3.87L8 6.091l1.06-1.06L5.03 1zM11.03 15.121l4.03-4.03-1.06-1.06-2.22 2.219V6.06h-1.5v6.19l-2.22-2.22L7 11.091l4.03 4.03z",
      fill: "currentColor"
    })
  });
}
function ArrowsUpDownIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgArrowsUpDownIconV2 : SvgArrowsUpDownIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgBarChartIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M12 5c.77 0 1.4.63 1.4 1.4v11.2c0 .77-.63 1.4-1.4 1.4-.77 0-1.4-.63-1.4-1.4V6.4c0-.77.63-1.4 1.4-1.4zM6.4 9.2h.2c.77 0 1.4.63 1.4 1.4v7c0 .77-.63 1.4-1.4 1.4h-.2c-.77 0-1.4-.63-1.4-1.4v-7c0-.77.63-1.4 1.4-1.4zM19 14.4c0-.77-.63-1.4-1.4-1.4-.77 0-1.4.63-1.4 1.4v3.2c0 .77.63 1.4 1.4 1.4.77 0 1.4-.63 1.4-1.4v-3.2z",
      fill: "currentColor"
    })
  });
}
function SvgBarChartIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M1 1v13.25c0 .414.336.75.75.75H15v-1.5H2.5V1H1z",
      fill: "currentColor"
    }), jsx("path", {
      d: "M7 1v11h1.5V1H7zM10 5v7h1.5V5H10zM4 5v7h1.5V5H4zM13 12V8h1.5v4H13z",
      fill: "currentColor"
    })]
  });
}
function BarChartIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgBarChartIconV2 : SvgBarChartIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgBeakerIconV1(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M9 2h7v8.37l5.43 7.347c1.463 1.98.05 4.783-2.413 4.783H5.982c-2.462 0-3.876-2.803-2.412-4.783L9 10.37V2zm2 9.03l-5.822 7.876a1 1 0 00.804 1.594h13.035a1 1 0 00.804-1.594L14 11.029V4h-3v7.03z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M8 3a1 1 0 011-1h7a1 1 0 010 2H9a1 1 0 01-1-1z",
      fill: "currentColor"
    }), jsx("path", {
      d: "M4 19l2-3h13l2 3-1.5 2.5h-14L4 19z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M5.732 15.5h13.536l2.324 3.486L19.783 22H5.217l-1.808-3.014L5.732 15.5zm.536 1l-1.676 2.514L5.783 21h13.434l1.192-1.986-1.676-2.514H6.268z",
      fill: "currentColor"
    })]
  });
}
function SvgBeakerIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M5.75 1a.75.75 0 00-.75.75v6.089c0 .38-.173.739-.47.976l-2.678 2.143A2.27 2.27 0 003.27 15h9.46a2.27 2.27 0 001.418-4.042L11.47 8.815A1.25 1.25 0 0111 7.839V1.75a.75.75 0 00-.75-.75h-4.5zm.75 6.839V2.5h3v5.339c0 .606.2 1.188.559 1.661H5.942A2.75 2.75 0 006.5 7.839zM4.2 11L2.79 12.13a.77.77 0 00.48 1.37h9.461a.77.77 0 00.481-1.37L11.8 11H4.201z",
      fill: "currentColor"
    })
  });
}
function BeakerIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgBeakerIconV2 : SvgBeakerIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgBookIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M6 2h12c1.1 0 2 .9 2 2v16c0 1.1-.9 2-2 2H6c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2zm5 2H6v8l2.5-1.5L11 12V4z",
      fill: "currentColor"
    })
  });
}
function SvgBookIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M2.75 1a.75.75 0 00-.75.75v13.5c0 .414.336.75.75.75h10.5a.75.75 0 00.75-.75V1.75a.75.75 0 00-.75-.75H2.75zM7.5 2.5h-4v6.055l1.495-1.36a.75.75 0 011.01 0L7.5 8.555V2.5zm-4 8.082l2-1.818 2.245 2.041A.75.75 0 009 10.25V2.5h3.5v12h-9v-3.918z",
      fill: "currentColor"
    })
  });
}
function BookIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgBookIconV2 : SvgBookIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgBookmarkFillIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M17 3H7c-1.1 0-2 .9-2 2v16l7-3 7 3V5c0-1.1-.9-2-2-2z",
      fill: "currentColor"
    })
  });
}
function SvgBookmarkFillIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M2.75 0A.75.75 0 002 .75v14.5a.75.75 0 001.28.53L8 11.06l4.72 4.72a.75.75 0 001.28-.53V.75a.75.75 0 00-.75-.75H2.75z",
      fill: "currentColor"
    })
  });
}
function BookmarkFillIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgBookmarkFillIconV2 : SvgBookmarkFillIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgBookmarkIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M7 3h10c1.1 0 2 .9 2 2v16l-7-3-7 3V5c0-1.1.9-2 2-2zm5 12.82L17 18V6c0-.55-.45-1-1-1H8c-.55 0-1 .45-1 1v12l5-2.18z",
      fill: "currentColor"
    })
  });
}
function SvgBookmarkIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M2 .75A.75.75 0 012.75 0h10.5a.75.75 0 01.75.75v14.5a.75.75 0 01-1.28.53L8 11.06l-4.72 4.72A.75.75 0 012 15.25V.75zm1.5.75v11.94l3.97-3.97a.75.75 0 011.06 0l3.97 3.97V1.5h-9z",
      fill: "currentColor"
    })
  });
}
function BookmarkIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgBookmarkIconV2 : SvgBookmarkIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgBriefcaseFillIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M16 6.5h4c1.11 0 2 .89 2 2v11c0 1.11-.89 2-2 2H4c-1.11 0-2-.89-2-2l.01-11c0-1.11.88-2 1.99-2h4v-2c0-1.11.89-2 2-2h4c1.11 0 2 .89 2 2v2zm-6 0h4v-2h-4v2z",
      fill: "currentColor"
    })
  });
}
function SvgBriefcaseFillIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M5 4V2.75C5 1.784 5.784 1 6.75 1h2.5c.966 0 1.75.784 1.75 1.75V4h3.25a.75.75 0 01.75.75v9.5a.75.75 0 01-.75.75H1.75a.75.75 0 01-.75-.75v-9.5A.75.75 0 011.75 4H5zm1.5-1.25a.25.25 0 01.25-.25h2.5a.25.25 0 01.25.25V4h-3V2.75zm-4 5.423V6.195A7.724 7.724 0 008 8.485c2.15 0 4.095-.875 5.5-2.29v1.978A9.211 9.211 0 018 9.985a9.21 9.21 0 01-5.5-1.812z",
      fill: "currentColor"
    })
  });
}
function BriefcaseFillIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgBriefcaseFillIconV2 : SvgBriefcaseFillIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgBriefcaseIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M22 8.5c0-1.11-.89-2-2-2h-4v-2c0-1.11-.89-2-2-2h-4c-1.11 0-2 .89-2 2v2H4c-1.11 0-1.99.89-1.99 2L2 19.5c0 1.11.89 2 2 2h16c1.11 0 2-.89 2-2v-11zm-8-2v-2h-4v2h4zm-10 3v9c0 .55.45 1 1 1h14c.55 0 1-.45 1-1v-9c0-.55-.45-1-1-1H5c-.55 0-1 .45-1 1z",
      fill: "currentColor"
    })
  });
}
function SvgBriefcaseIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M1.75 4H5V2.75C5 1.784 5.784 1 6.75 1h2.5c.966 0 1.75.784 1.75 1.75V4h3.25a.75.75 0 01.75.75v9.5a.75.75 0 01-.75.75H1.75a.75.75 0 01-.75-.75v-9.5A.75.75 0 011.75 4zm5-1.5a.25.25 0 00-.25.25V4h3V2.75a.25.25 0 00-.25-.25h-2.5zM2.5 8.173V13.5h11V8.173A9.211 9.211 0 018 9.985a9.21 9.21 0 01-5.5-1.812zm0-1.978A7.724 7.724 0 008 8.485c2.15 0 4.095-.875 5.5-2.29V5.5h-11v.695z",
      fill: "currentColor"
    })
  });
}
function BriefcaseIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgBriefcaseIconV2 : SvgBriefcaseIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgCalendarEventIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M16 3v1H8V3c0-.55-.45-1-1-1s-1 .45-1 1v1H5c-1.11 0-1.99.9-1.99 2L3 20a2 2 0 002 2h14c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2h-1V3c0-.55-.45-1-1-1s-1 .45-1 1zm0 10h-3c-.55 0-1 .45-1 1v3c0 .55.45 1 1 1h3c.55 0 1-.45 1-1v-3c0-.55-.45-1-1-1zM6 20h12c.55 0 1-.45 1-1V9H5v10c0 .55.45 1 1 1z",
      fill: "currentColor"
    })
  });
}
function SvgCalendarEventIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M8.5 10.25a1.75 1.75 0 113.5 0 1.75 1.75 0 01-3.5 0z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M10 2H6V0H4.5v2H1.75a.75.75 0 00-.75.75v11.5c0 .414.336.75.75.75h12.5a.75.75 0 00.75-.75V2.75a.75.75 0 00-.75-.75H11.5V0H10v2zM2.5 3.5v2h11v-2h-11zm0 10V7h11v6.5h-11z",
      fill: "currentColor"
    })]
  });
}
function CalendarEventIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgCalendarEventIconV2 : SvgCalendarEventIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgCalendarIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M19 3h1c1.1 0 2 .9 2 2v16c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V5c0-1.1.9-2 2-2h1V2c0-.55.45-1 1-1s1 .45 1 1v1h10V2c0-.55.45-1 1-1s1 .45 1 1v1zM5 21h14c.55 0 1-.45 1-1V8H4v12c0 .55.45 1 1 1z",
      fill: "currentColor"
    })
  });
}
function SvgCalendarIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M4.5 0v2H1.75a.75.75 0 00-.75.75v11.5c0 .414.336.75.75.75h12.5a.75.75 0 00.75-.75V2.75a.75.75 0 00-.75-.75H11.5V0H10v2H6V0H4.5zm9 3.5v2h-11v-2h11zM2.5 7v6.5h11V7h-11z",
      fill: "currentColor"
    })
  });
}
function CalendarIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgCalendarIconV2 : SvgCalendarIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgCaretDownSquareIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M5 21h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2zM6 5h12c.55 0 1 .45 1 1v12c0 .55-.45 1-1 1H6c-.55 0-1-.45-1-1V6c0-.55.45-1 1-1zm6.703 9.004l2.59-2.59c.63-.63.19-1.71-.7-1.71h-5.18c-.89 0-1.34 1.08-.71 1.71l2.59 2.59c.39.39 1.02.39 1.41 0z",
      fill: "currentColor"
    })
  });
}
function SvgCaretDownSquareIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M8 10a.75.75 0 01-.59-.286l-2.164-2.75a.75.75 0 01.589-1.214h4.33a.75.75 0 01.59 1.214l-2.166 2.75A.75.75 0 018 10z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M1.75 1a.75.75 0 00-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 00.75-.75V1.75a.75.75 0 00-.75-.75H1.75zm.75 12.5v-11h11v11h-11z",
      fill: "currentColor"
    })]
  });
}
function CaretDownSquareIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgCaretDownSquareIconV2 : SvgCaretDownSquareIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgCaretUpSquareIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-1 16H6c-.55 0-1-.45-1-1V6c0-.55.45-1 1-1h12c.55 0 1 .45 1 1v12c0 .55-.45 1-1 1zm-6.703-9.004l-2.59 2.59c-.63.63-.19 1.71.7 1.71h5.18c.89 0 1.34-1.08.71-1.71l-2.59-2.59a.996.996 0 00-1.41 0z",
      fill: "currentColor"
    })
  });
}
function SvgCaretUpSquareIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M8 5.75a.75.75 0 01.59.286l2.164 2.75A.75.75 0 0110.165 10h-4.33a.75.75 0 01-.59-1.214l2.166-2.75A.75.75 0 018 5.75z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M1.75 1a.75.75 0 00-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 00.75-.75V1.75a.75.75 0 00-.75-.75H1.75zm.75 12.5v-11h11v11h-11z",
      fill: "currentColor"
    })]
  });
}
function CaretUpSquareIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgCaretUpSquareIconV2 : SvgCaretUpSquareIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgCheckCircleFillIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M2 12C2 6.48 6.48 2 12 2s10 4.48 10 10-4.48 10-10 10S2 17.52 2 12zm3.7.7l3.59 3.59c.39.39 1.03.39 1.41 0l7.59-7.59a.996.996 0 10-1.41-1.41L10 14.17l-2.89-2.88A.996.996 0 105.7 12.7z",
      fill: "currentColor"
    })
  });
}
function SvgCheckCircleFillIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M0 8a8 8 0 1116 0A8 8 0 010 8zm11.53-1.47l-1.06-1.06L7 8.94 5.53 7.47 4.47 8.53l2 2 .53.53.53-.53 4-4z",
      fill: "currentColor"
    })
  });
}
function CheckCircleFillIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgCheckCircleFillIconV2 : SvgCheckCircleFillIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgCheckCircleIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-2-5.83l5.88-5.88c.39-.39 1.03-.39 1.42 0 .39.39.39 1.02 0 1.41l-6.59 6.59a.996.996 0 01-1.41 0L6.71 13.7a.996.996 0 111.41-1.41L10 14.17z",
      fill: "currentColor"
    })
  });
}
function SvgCheckCircleIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M11.53 6.53L7 11.06 4.47 8.53l1.06-1.06L7 8.94l3.47-3.47 1.06 1.06z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M0 8a8 8 0 1116 0A8 8 0 010 8zm8-6.5a6.5 6.5 0 100 13 6.5 6.5 0 000-13z",
      fill: "currentColor"
    })]
  });
}
function CheckCircleIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgCheckCircleIconV2 : SvgCheckCircleIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgCheckIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M8.795 15.875l-3.47-3.47a.996.996 0 00-1.41 0 .996.996 0 000 1.41l4.18 4.18c.39.39 1.02.39 1.41 0l10.58-10.58a.996.996 0 10-1.41-1.41l-9.88 9.87z",
      fill: "currentColor"
    })
  });
}
function SvgCheckIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M15.06 3.56l-9.53 9.531L1 8.561 2.06 7.5l3.47 3.47L14 2.5l1.06 1.06z",
      fill: "currentColor"
    })
  });
}
function CheckIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgCheckIconV2 : SvgCheckIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgCheckLineIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M8.19 13.607a2 2 0 002.82.01l7-6.93c.55-.54.55-1.42.01-1.96l-.04-.04c-.54-.54-1.41-.54-1.95 0l-6.43 6.43-1.65-1.65c-.52-.52-1.38-.54-1.92-.02-.57.53-.58 1.42-.03 1.97l2.19 2.19zm9.81 4.11H6c-.55 0-1 .45-1 1s.45 1 1 1h12c.55 0 1-.45 1-1s-.45-1-1-1z",
      fill: "currentColor"
    })
  });
}
function SvgCheckLineIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M15.06 2.06L14 1 5.53 9.47 2.06 6 1 7.06l4.53 4.531 9.53-9.53zM1.03 15.03h14v-1.5h-14v1.5z",
      fill: "currentColor"
    })
  });
}
function CheckLineIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgCheckLineIconV2 : SvgCheckLineIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgChecklistIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M4.5 7.914l5.207-5.207-1.414-1.414L4.5 5.086 2.707 3.293 1.293 4.707 4.5 7.914zM11 4h11v2H11V4zM11 11h11v2H11v-2zM22 18H2v2h20v-2zM9.707 9.707L4.5 14.914l-3.207-3.207 1.414-1.414L4.5 12.086l3.793-3.793 1.414 1.414z",
      fill: "currentColor"
    })
  });
}
function SvgChecklistIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M5.5 2l1.06 1.06-3.53 3.531L1 4.561 2.06 3.5l.97.97L5.5 2zM15.03 4.53h-7v-1.5h7v1.5zM1.03 14.53v-1.5h14v1.5h-14zM8.03 9.53h7v-1.5h-7v1.5zM6.56 8.06L5.5 7 3.03 9.47l-.97-.97L1 9.56l2.03 2.031 3.53-3.53z",
      fill: "currentColor"
    })
  });
}
function ChecklistIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgChecklistIconV2 : SvgChecklistIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgChevronDoubleDownIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M9.54 12.71L12 15.17l2.47-2.46a.996.996 0 111.41 1.41l-3.17 3.18a.996.996 0 01-1.41 0l-3.17-3.18a.996.996 0 111.41-1.41zM9.53 6.7l2.46 2.46 2.47-2.45a.996.996 0 111.41 1.41l-3.17 3.17a.996.996 0 01-1.41 0L8.12 8.11A.996.996 0 119.53 6.7z",
      fill: "currentColor"
    })
  });
}
function SvgChevronDoubleDownIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M10.947 7.954L8 10.891 5.056 7.954 3.997 9.016l4.004 3.993 4.005-3.993-1.06-1.062z",
      fill: "currentColor"
    }), jsx("path", {
      d: "M10.947 3.994L8 6.931 5.056 3.994 3.997 5.056 8.001 9.05l4.005-3.993-1.06-1.062z",
      fill: "currentColor"
    })]
  });
}
function ChevronDoubleDownIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgChevronDoubleDownIconV2 : SvgChevronDoubleDownIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgChevronDoubleLeftIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M11.29 9.54L8.83 12l2.46 2.47a.996.996 0 11-1.41 1.41L6.7 12.71a.996.996 0 010-1.41l3.18-3.17a.996.996 0 111.41 1.41zm6.01-.01l-2.46 2.46 2.45 2.47a.996.996 0 11-1.41 1.41l-3.17-3.17a.996.996 0 010-1.41l3.18-3.17a.996.996 0 111.41 1.41z",
      fill: "currentColor"
    })
  });
}
function SvgChevronDoubleLeftIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M8.047 10.944L5.11 8l2.937-2.944-1.062-1.06L2.991 8l3.994 4.003 1.062-1.06z",
      fill: "currentColor"
    }), jsx("path", {
      d: "M12.008 10.944L9.07 8l2.938-2.944-1.062-1.06L6.952 8l3.994 4.003 1.062-1.06z",
      fill: "currentColor"
    })]
  });
}
function ChevronDoubleLeftIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgChevronDoubleLeftIconV2 : SvgChevronDoubleLeftIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgChevronDoubleRightIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M12.71 14.46L15.17 12l-2.46-2.47a.996.996 0 111.41-1.41l3.18 3.17a.996.996 0 010 1.41l-3.18 3.17a.996.996 0 11-1.41-1.41zm-6.01.01l2.46-2.46-2.45-2.47a.996.996 0 111.41-1.41l3.17 3.17c.39.39.39 1.02 0 1.41l-3.18 3.17a.996.996 0 11-1.41-1.41z",
      fill: "currentColor"
    })
  });
}
function SvgChevronDoubleRightIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M7.954 5.056l2.937 2.946-2.937 2.945 1.062 1.059L13.01 8 9.016 3.998l-1.062 1.06z",
      fill: "currentColor"
    }), jsx("path", {
      d: "M3.994 5.056l2.937 2.946-2.937 2.945 1.062 1.059L9.05 8 5.056 3.998l-1.062 1.06z",
      fill: "currentColor"
    })]
  });
}
function ChevronDoubleRightIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgChevronDoubleRightIconV2 : SvgChevronDoubleRightIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgChevronDoubleUpIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M14.46 11.29L12 8.83l-2.47 2.46a.996.996 0 11-1.41-1.41l3.17-3.18a.996.996 0 011.41 0l3.17 3.18a.996.996 0 11-1.41 1.41zm.01 6.01l-2.46-2.46-2.47 2.45a.996.996 0 11-1.41-1.41l3.17-3.17a.996.996 0 011.41 0l3.17 3.18a.996.996 0 11-1.41 1.41z",
      fill: "currentColor"
    })
  });
}
function SvgChevronDoubleUpIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M5.056 8.047L8 5.11l2.944 2.937 1.06-1.062L8 2.991 3.997 6.985l1.059 1.062z",
      fill: "currentColor"
    }), jsx("path", {
      d: "M5.056 12.008L8 9.07l2.944 2.937 1.06-1.062L8 6.952l-4.003 3.994 1.059 1.062z",
      fill: "currentColor"
    })]
  });
}
function ChevronDoubleUpIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgChevronDoubleUpIconV2 : SvgChevronDoubleUpIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgChevronDownIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M15.875 9l-3.88 3.88L8.115 9a.996.996 0 10-1.41 1.41l4.59 4.59c.39.39 1.02.39 1.41 0l4.59-4.59a.996.996 0 000-1.41c-.39-.38-1.03-.39-1.42 0z",
      fill: "currentColor"
    })
  });
}
function SvgChevronDownIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M8 8.917L10.947 6 12 7.042 8 11 4 7.042 5.053 6 8 8.917z",
      fill: "currentColor"
    })
  });
}
function ChevronDownIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgChevronDownIconV2 : SvgChevronDownIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgChevronLeftIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M15 15.875l-3.88-3.88L15 8.115a.996.996 0 10-1.41-1.41L9 11.295a.996.996 0 000 1.41l4.59 4.59c.39.39 1.02.39 1.41 0 .38-.39.39-1.03 0-1.42z",
      fill: "currentColor"
    })
  });
}
function SvgChevronLeftIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M7.083 8L10 10.947 8.958 12 5 8l3.958-4L10 5.053 7.083 8z",
      fill: "currentColor"
    })
  });
}
function ChevronLeftIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgChevronLeftIconV2 : SvgChevronLeftIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgChevronRightIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M9 15.875l3.88-3.88L9 8.115a.996.996 0 111.41-1.41l4.59 4.59c.39.39.39 1.02 0 1.41l-4.59 4.59a.996.996 0 01-1.41 0c-.38-.39-.39-1.03 0-1.42z",
      fill: "currentColor"
    })
  });
}
function SvgChevronRightIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M8.917 8L6 5.053 7.042 4 11 8l-3.958 4L6 10.947 8.917 8z",
      fill: "currentColor"
    })
  });
}
function ChevronRightIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgChevronRightIconV2 : SvgChevronRightIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgChevronUpIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M11.295 9l-4.59 4.59A.996.996 0 108.115 15l3.89-3.88 3.88 3.88a.996.996 0 101.41-1.41L12.705 9a.996.996 0 00-1.41 0z",
      fill: "currentColor"
    })
  });
}
function SvgChevronUpIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M8 7.083L5.053 10 4 8.958 8 5l4 3.958L10.947 10 8 7.083z",
      fill: "currentColor"
    })
  });
}
function ChevronUpIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgChevronUpIconV2 : SvgChevronUpIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgCircleIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M12 4a8 8 0 100 16 8 8 0 000-16z",
      fill: "currentColor"
    })
  });
}
function SvgCircleIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M12.5 8a4.5 4.5 0 11-9 0 4.5 4.5 0 019 0z",
      fill: "currentColor"
    })
  });
}
function CircleIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgCircleIconV2 : SvgCircleIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgClipboardIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M19 3h-4.18C14.4 1.84 13.3 1 12 1c-1.3 0-2.4.84-2.82 2H5c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-7 0c.55 0 1 .45 1 1s-.45 1-1 1-1-.45-1-1 .45-1 1-1zM5 20c0 .55.45 1 1 1h12c.55 0 1-.45 1-1V6c0-.55-.45-1-1-1h-1v1c0 1.1-.9 2-2 2H9c-1.1 0-2-.9-2-2V5H6c-.55 0-1 .45-1 1v14z",
      fill: "currentColor"
    })
  });
}
function SvgClipboardIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M5.5 0a.75.75 0 00-.75.75V1h-2a.75.75 0 00-.75.75v12.5c0 .414.336.75.75.75h10.5a.75.75 0 00.75-.75V1.75a.75.75 0 00-.75-.75h-2V.75A.75.75 0 0010.5 0h-5zm5.75 2.5v.75a.75.75 0 01-.75.75h-5a.75.75 0 01-.75-.75V2.5H3.5v11h9v-11h-1.25zm-5 0v-1h3.5v1h-3.5z",
      fill: "currentColor"
    })
  });
}
function ClipboardIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgClipboardIconV2 : SvgClipboardIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgClockIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M11.99 2C6.47 2 2 6.48 2 12s4.47 10 9.99 10C17.52 22 22 17.52 22 12S17.52 2 11.99 2zM12 20c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8zm-.28-13h.06c.4 0 .72.32.72.72v4.54l3.87 2.3c.35.2.46.65.25.99-.2.34-.64.44-.98.24l-4.15-2.49a.99.99 0 01-.49-.86V7.72c0-.4.32-.72.72-.72z",
      fill: "currentColor"
    })
  });
}
function SvgClockIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M7.25 4v4c0 .199.079.39.22.53l2 2 1.06-1.06-1.78-1.78V4h-1.5z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M8 0a8 8 0 100 16A8 8 0 008 0zM1.5 8a6.5 6.5 0 1113 0 6.5 6.5 0 01-13 0z",
      fill: "currentColor"
    })]
  });
}
function ClockIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgClockIconV2 : SvgClockIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgCloseIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M7.707 6.293a1 1 0 00-1.414 1.414L10.586 12l-4.293 4.293a1 1 0 101.414 1.414L12 13.414l4.293 4.293a1 1 0 001.414-1.414L13.414 12l4.293-4.293a1 1 0 00-1.414-1.414L12 10.586 7.707 6.293z",
      fill: "currentColor"
    })
  });
}
function SvgCloseIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M6.97 8.03L2 3.06 3.06 2l4.97 4.97L13 2l1.06 1.06-4.969 4.97 4.97 4.97L13 14.06 8.03 9.092l-4.97 4.97L2 13l4.97-4.97z",
      fill: "currentColor"
    })
  });
}
function CloseIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgCloseIconV2 : SvgCloseIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgCloudDownloadIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M12 4a7.49 7.49 0 017.35 6.04c2.6.18 4.65 2.32 4.65 4.96 0 2.76-2.24 5-5 5H6c-3.31 0-6-2.69-6-6 0-3.09 2.34-5.64 5.35-5.96A7.496 7.496 0 0112 4zm.35 13.65L17 13h-3V9h-4v4H7l4.64 4.65c.2.2.51.2.71 0z",
      fill: "currentColor"
    })
  });
}
function SvgCloudDownloadIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M8 2a4.752 4.752 0 00-4.606 3.586 4.251 4.251 0 00.427 8.393A.75.75 0 004 14v-1.511a2.75 2.75 0 01.077-5.484.75.75 0 00.697-.657 3.25 3.25 0 016.476.402v.5c0 .414.336.75.75.75h.25a2.25 2.25 0 11-.188 4.492.75.75 0 00-.062-.002V14a.757.757 0 00.077-.004 3.75 3.75 0 00.668-7.464A4.75 4.75 0 008 2z",
      fill: "currentColor"
    }), jsx("path", {
      d: "M7.25 11.19L5.03 8.97l-1.06 1.06L8 14.06l4.03-4.03-1.06-1.06-2.22 2.22V6h-1.5v5.19z",
      fill: "currentColor"
    })]
  });
}
function CloudDownloadIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgCloudDownloadIconV2 : SvgCloudDownloadIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgCloudIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M12 4a7.49 7.49 0 017.35 6.04c2.6.18 4.65 2.32 4.65 4.96 0 2.76-2.24 5-5 5H6c-3.31 0-6-2.69-6-6 0-3.09 2.34-5.64 5.35-5.96A7.496 7.496 0 0112 4zM6 18h13c1.66 0 3-1.34 3-3s-1.34-3-3-3h-1.5v-.5C17.5 8.46 15.04 6 12 6c-2.52 0-4.63 1.69-5.29 4H6c-2.21 0-4 1.79-4 4s1.79 4 4 4z",
      fill: "currentColor"
    })
  });
}
function SvgCloudIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M3.394 5.586a4.752 4.752 0 019.351.946 3.75 3.75 0 01-.668 7.464A.757.757 0 0112 14H4a.75.75 0 01-.179-.021 4.25 4.25 0 01-.427-8.393zm.72 6.914h7.762a.745.745 0 01.186-.008A2.25 2.25 0 1012.25 8H12a.75.75 0 01-.75-.75v-.5a3.25 3.25 0 00-6.476-.402.75.75 0 01-.697.657 2.75 2.75 0 00-.024 5.488.74.74 0 01.062.007z",
      fill: "currentColor"
    })
  });
}
function CloudIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgCloudIconV2 : SvgCloudIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgCloudOffIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M23.998 14.209c0-2.64-2.05-4.78-4.65-4.96a7.49 7.49 0 00-7.35-6.04c-1.33 0-2.57.36-3.65.97l1.49 1.49c.67-.29 1.39-.46 2.16-.46 3.04 0 5.5 2.46 5.5 5.5v.5h1.5a2.996 2.996 0 011.79 5.4l1.41 1.41c1.09-.92 1.8-2.27 1.8-3.81zM3.708 3.769a.996.996 0 000 1.41l2.06 2.06h-.42a5.99 5.99 0 00-5.29 6.79c.4 3.02 3.13 5.18 6.16 5.18h11.51l1.29 1.29a.996.996 0 101.41-1.41L5.118 3.769a.996.996 0 00-1.41 0zm-1.71 9.44c0 2.21 1.79 4 4 4h9.73l-8-8h-1.73c-2.21 0-4 1.79-4 4z",
      fill: "currentColor"
    })
  });
}
function SvgCloudOffIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M13.97 14.53L2.47 3.03l-1 1 1.628 1.628a4.252 4.252 0 00.723 8.32A.75.75 0 004 14h7.44l1.53 1.53 1-1zM4.077 7.005a.748.748 0 00.29-.078L9.939 12.5H4.115a.74.74 0 00-.062-.007 2.75 2.75 0 01.024-5.488z",
      fill: "currentColor"
    }), jsx("path", {
      d: "M4.8 3.24a4.75 4.75 0 017.945 3.293 3.75 3.75 0 011.928 6.58l-1.067-1.067A2.25 2.25 0 0012.25 8H12a.75.75 0 01-.75-.75v-.5a3.25 3.25 0 00-5.388-2.448L4.8 3.239z",
      fill: "currentColor"
    })]
  });
}
function CloudOffIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgCloudOffIconV2 : SvgCloudOffIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgCloudUploadIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M12 4a7.49 7.49 0 017.35 6.04c2.6.18 4.65 2.32 4.65 4.96 0 2.76-2.24 5-5 5H6c-3.31 0-6-2.69-6-6 0-3.09 2.34-5.64 5.35-5.96A7.496 7.496 0 0112 4zm2 13v-4h3l-4.64-4.65c-.2-.2-.51-.2-.71 0L7 13h3v4h4z",
      fill: "currentColor"
    })
  });
}
function SvgCloudUploadIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M8 2a4.752 4.752 0 00-4.606 3.586 4.251 4.251 0 00.427 8.393A.75.75 0 004 14v-1.511a2.75 2.75 0 01.077-5.484.75.75 0 00.697-.657 3.25 3.25 0 016.476.402v.5c0 .414.336.75.75.75h.25a2.25 2.25 0 11-.188 4.492.75.75 0 00-.062-.002V14a.757.757 0 00.077-.004 3.75 3.75 0 00.668-7.464A4.75 4.75 0 008 2z",
      fill: "currentColor"
    }), jsx("path", {
      d: "M8.75 8.81l2.22 2.22 1.06-1.06L8 5.94 3.97 9.97l1.06 1.06 2.22-2.22V14h1.5V8.81z",
      fill: "currentColor"
    })]
  });
}
function CloudUploadIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgCloudUploadIconV2 : SvgCloudUploadIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgCodeIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M19.2 12l-3.9 3.9a.984.984 0 000 1.4c.39.39 1.01.39 1.4 0l4.59-4.6a.996.996 0 000-1.41L16.7 6.7a.984.984 0 00-1.4 0 .984.984 0 000 1.4l3.9 3.9zM4.8 12l3.9 3.9c.39.39.39 1.01 0 1.4a.984.984 0 01-1.4 0l-4.59-4.6a.996.996 0 010-1.41L7.3 6.7a.984.984 0 011.4 0c.39.39.39 1.01 0 1.4L4.8 12z",
      fill: "currentColor"
    })
  });
}
function SvgCodeIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 17 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M4.03 12.06L5.091 11l-2.97-2.97 2.97-2.97L4.031 4 0 8.03l4.03 4.03zM12.091 4l4.03 4.03-4.03 4.03-1.06-1.06L14 8.03l-2.97-2.97L12.091 4z",
      fill: "currentColor"
    })
  });
}
function CodeIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgCodeIconV2 : SvgCodeIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgCopyIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M15.5 1h-11c-1.1 0-2 .9-2 2v13c0 .55.45 1 1 1s1-.45 1-1V4c0-.55.45-1 1-1h10c.55 0 1-.45 1-1s-.45-1-1-1zm.59 4.59l4.83 4.83c.37.37.58.88.58 1.41V21c0 1.1-.9 2-2 2H8.49c-1.1 0-1.99-.9-1.99-2l.01-14c0-1.1.89-2 1.99-2h6.17c.53 0 1.04.21 1.42.59zM20 12h-4.5c-.55 0-1-.45-1-1V6.5L20 12z",
      fill: "currentColor"
    })
  });
}
function SvgCopyIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M1.75 1a.75.75 0 00-.75.75v8.5c0 .414.336.75.75.75H5v3.25c0 .414.336.75.75.75h8.5a.75.75 0 00.75-.75v-8.5a.75.75 0 00-.75-.75H11V1.75a.75.75 0 00-.75-.75h-8.5zM9.5 5V2.5h-7v7H5V5.75A.75.75 0 015.75 5H9.5zm-3 8.5v-7h7v7h-7z",
      fill: "currentColor"
    })
  });
}
function CopyIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgCopyIconV2 : SvgCopyIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgDIconV1(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M11.487 16c2.66 0 4.27-1.645 4.27-4.372 0-2.719-1.61-4.355-4.244-4.355h-3.12V16h3.094zm-1.248-1.581V8.854h1.176c1.636 0 2.501.835 2.501 2.774 0 1.947-.865 2.791-2.506 2.791H10.24z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M17 3H7a4 4 0 00-4 4v10a4 4 0 004 4h10a4 4 0 004-4V7a4 4 0 00-4-4zM7 1a6 6 0 00-6 6v10a6 6 0 006 6h10a6 6 0 006-6V7a6 6 0 00-6-6H7z",
      fill: "currentColor"
    })]
  });
}
function SvgDIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M5.75 4.5a.75.75 0 00-.75.75v5.5c0 .414.336.75.75.75h2a3.5 3.5 0 100-7h-2zM6.5 10V6h1.25a2 2 0 110 4H6.5z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M1.75 1a.75.75 0 00-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 00.75-.75V1.75a.75.75 0 00-.75-.75H1.75zm.75 12.5v-11h11v11h-11z",
      fill: "currentColor"
    })]
  });
}
function DIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgDIconV2 : SvgDIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgDangerFillIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 11c-.55 0-1-.45-1-1V8c0-.55.45-1 1-1s1 .45 1 1v4c0 .55-.45 1-1 1zm-1 2v2h2v-2h-2z",
      fill: "currentColor"
    })
  });
}
function SvgDangerFillIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M15.78 11.533l-4.242 4.243a.75.75 0 01-.53.22H4.996a.75.75 0 01-.53-.22L.224 11.533a.75.75 0 01-.22-.53v-6.01a.75.75 0 01.22-.53L4.467.22a.75.75 0 01.53-.22h6.01a.75.75 0 01.53.22l4.243 4.242c.141.141.22.332.22.53v6.011a.75.75 0 01-.22.53zm-8.528-.785a.75.75 0 101.5 0 .75.75 0 00-1.5 0zm1.5-5.75v4h-1.5v-4h1.5z",
      fill: "currentColor"
    })
  });
}
function DangerFillIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgDangerFillIconV2 : SvgDangerFillIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgDangerIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M2 12C2 6.48 6.47 2 11.99 2 17.52 2 22 6.48 22 12s-4.48 10-10.01 10C6.47 22 2 17.52 2 12zm11-4c0-.55-.45-1-1-1s-1 .45-1 1v4c0 .55.45 1 1 1s1-.45 1-1V8zm-1 12c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8zm-1-5v2h2v-2h-2z",
      fill: "currentColor"
    })
  });
}
function SvgDangerIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M7.248 10.748a.75.75 0 101.5 0 .75.75 0 00-1.5 0zM8.748 4.998v4h-1.5v-4h1.5z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M11.533 15.776l4.243-4.243a.75.75 0 00.22-.53v-6.01a.75.75 0 00-.22-.53L11.533.22a.75.75 0 00-.53-.22h-6.01a.75.75 0 00-.53.22L.22 4.462a.75.75 0 00-.22.53v6.011c0 .199.079.39.22.53l4.242 4.243c.141.14.332.22.53.22h6.011a.75.75 0 00.53-.22zm2.963-10.473v5.39l-3.804 3.803H5.303L1.5 10.692V5.303L5.303 1.5h5.39l3.803 3.803z",
      fill: "currentColor"
    })]
  });
}
function DangerIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgDangerIconV2 : SvgDangerIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgDashIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M18 13H6c-.55 0-1-.45-1-1s.45-1 1-1h12c.55 0 1 .45 1 1s-.45 1-1 1z",
      fill: "currentColor"
    })
  });
}
function SvgDashIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M15 8.75H1v-1.5h14v1.5z",
      fill: "currentColor"
    })
  });
}
function DashIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgDashIconV2 : SvgDashIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgDashboardIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M3 13h8V3H3v10zm0 8h8v-6H3v6zm10 0h8V11h-8v10zm0-18v6h8V3h-8z",
      fill: "currentColor"
    })
  });
}
function SvgDashboardIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M1 1.75A.75.75 0 011.75 1h12.5a.75.75 0 01.75.75v12.5a.75.75 0 01-.75.75H1.75a.75.75 0 01-.75-.75V1.75zm1.5 8.75v3h4.75v-3H2.5zm0-1.5h4.75V2.5H2.5V9zm6.25-6.5v3h4.75v-3H8.75zm0 11V7h4.75v6.5H8.75z",
      fill: "currentColor"
    })
  });
}
function DashboardIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgDashboardIconV2 : SvgDashboardIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgDataIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M21 15h-6v6h6v-6zm-8-2v10h10V13H13zM6 21a3 3 0 100-6 3 3 0 000 6zm0 2a5 5 0 100-10 5 5 0 000 10zM12 0L5.938 11h12.124L12 0zm0 4.144L9.324 9h5.352L12 4.144z",
      fill: "currentColor"
    })
  });
}
function SvgDataIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M8.646.368a.75.75 0 00-1.292 0l-3.25 5.5A.75.75 0 004.75 7h6.5a.75.75 0 00.646-1.132l-3.25-5.5zM8 2.224L9.936 5.5H6.064L8 2.224zM8.5 9.25a.75.75 0 01.75-.75h5a.75.75 0 01.75.75v5a.75.75 0 01-.75.75h-5a.75.75 0 01-.75-.75v-5zM10 10v3.5h3.5V10H10zM1 11.75a3.25 3.25 0 116.5 0 3.25 3.25 0 01-6.5 0zM4.25 10a1.75 1.75 0 100 3.5 1.75 1.75 0 000-3.5z",
      fill: "currentColor"
    })
  });
}
function DataIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgDataIconV2 : SvgDataIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgDatabaseIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M5 7V5c0-1.105 3.134-2 7-2s7 .895 7 2v2c0 1.105-3.134 2-7 2s-7-.895-7-2zM5 10v2.155c0 1.104 3.134 2 7 2s7-.896 7-2V10c-1.65.831-4.173 1.155-7 1.155S6.65 10.83 5 10zM5 17.31v-2.155c1.65.83 4.173 1.154 7 1.154s5.35-.323 7-1.154v2.154c0 1.105-3.134 2-7 2s-7-.895-7-2z",
      fill: "currentColor"
    })
  });
}
function SvgDatabaseIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M2.727 3.695c-.225.192-.227.298-.227.305 0 .007.002.113.227.305.223.19.59.394 1.108.58C4.865 5.256 6.337 5.5 8 5.5c1.663 0 3.135-.244 4.165-.615.519-.186.885-.39 1.108-.58.225-.192.227-.298.227-.305 0-.007-.002-.113-.227-.305-.223-.19-.59-.394-1.108-.58C11.135 2.744 9.663 2.5 8 2.5c-1.663 0-3.135.244-4.165.615-.519.186-.885.39-1.108.58zM13.5 5.94a6.646 6.646 0 01-.826.358C11.442 6.74 9.789 7 8 7c-1.79 0-3.442-.26-4.673-.703a6.641 6.641 0 01-.827-.358V8c0 .007.002.113.227.305.223.19.59.394 1.108.58C4.865 9.256 6.337 9.5 8 9.5c1.663 0 3.135-.244 4.165-.615.519-.186.885-.39 1.108-.58.225-.192.227-.298.227-.305V5.939zM15 8V4c0-.615-.348-1.1-.755-1.447-.41-.349-.959-.63-1.571-.85C11.442 1.26 9.789 1 8 1c-1.79 0-3.442.26-4.673.703-.613.22-1.162.501-1.572.85C1.348 2.9 1 3.385 1 4v8c0 .615.348 1.1.755 1.447.41.349.959.63 1.572.85C4.558 14.74 6.21 15 8 15c1.79 0 3.441-.26 4.674-.703.612-.22 1.161-.501 1.571-.85.407-.346.755-.832.755-1.447V8zm-1.5 1.939a6.654 6.654 0 01-.826.358C11.442 10.74 9.789 11 8 11c-1.79 0-3.442-.26-4.673-.703a6.649 6.649 0 01-.827-.358V12c0 .007.002.113.227.305.223.19.59.394 1.108.58 1.03.371 2.502.615 4.165.615 1.663 0 3.135-.244 4.165-.615.519-.186.885-.39 1.108-.58.225-.192.227-.298.227-.305V9.939z",
      fill: "currentColor"
    })
  });
}
function DatabaseIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgDatabaseIconV2 : SvgDatabaseIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgDownloadIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M15 9.5h1.59c.89 0 1.33 1.08.7 1.71L12.7 15.8a.996.996 0 01-1.41 0L6.7 11.21c-.63-.63-.18-1.71.71-1.71H9v-5c0-.55.45-1 1-1h4c.55 0 1 .45 1 1v5zm-9 11c-.55 0-1-.45-1-1s.45-1 1-1h12c.55 0 1 .45 1 1s-.45 1-1 1H6z",
      fill: "currentColor"
    })
  });
}
function SvgDownloadIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M1 13.5h14V15H1v-1.5zM12.53 6.53l-1.06-1.06-2.72 2.72V1h-1.5v7.19L4.53 5.47 3.47 6.53 8 11.06l4.53-4.53z",
      fill: "currentColor"
    })
  });
}
function DownloadIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgDownloadIconV2 : SvgDownloadIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgDragIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M9 4c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm-2 8c0-1.1.9-2 2-2s2 .9 2 2-.9 2-2 2-2-.9-2-2zm2 8c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zm8-14c0 1.1-.9 2-2 2s-2-.9-2-2 .9-2 2-2 2 .9 2 2zm-2 4c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm-2 8c0-1.1.9-2 2-2s2 .9 2 2-.9 2-2 2-2-.9-2-2z",
      fill: "currentColor"
    })
  });
}
function SvgDragIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M5.25 1a1.75 1.75 0 100 3.5 1.75 1.75 0 000-3.5zM10.75 1a1.75 1.75 0 100 3.5 1.75 1.75 0 000-3.5zM5.25 6.25a1.75 1.75 0 100 3.5 1.75 1.75 0 000-3.5zM10.75 6.25a1.75 1.75 0 100 3.5 1.75 1.75 0 000-3.5zM5.25 11.5a1.75 1.75 0 100 3.5 1.75 1.75 0 000-3.5zM10.75 11.5a1.75 1.75 0 100 3.5 1.75 1.75 0 000-3.5z",
      fill: "currentColor"
    })
  });
}
function DragIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgDragIconV2 : SvgDragIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgExpandLessIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M15.175 4.407a.996.996 0 01.695 1.713L12.7 9.29a.996.996 0 01-1.41 0L8.12 6.12c-.39-.39-.39-1.03 0-1.42a.996.996 0 011.41 0L12 7.17l2.47-2.47a.996.996 0 01.705-.293zM12 16.83L9.53 19.3a.996.996 0 01-1.41 0 .987.987 0 01.01-1.41l3.17-3.17a.996.996 0 011.41 0l3.17 3.17a.996.996 0 11-1.41 1.41L12 16.83z",
      fill: "currentColor"
    })
  });
}
function SvgExpandLessIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 17",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M12.06 1.06L11 0 8.03 2.97 5.06 0 4 1.06l4.03 4.031 4.03-4.03zM4 15l4.03-4.03L12.06 15 11 16.06l-2.97-2.969-2.97 2.97L4 15z",
      fill: "currentColor"
    })
  });
}
function ExpandLessIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgExpandLessIconV2 : SvgExpandLessIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgExpandMoreIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M14.46 8.29L12 5.83 9.53 8.29a.996.996 0 11-1.41-1.41l3.17-3.18a.996.996 0 011.41 0l3.17 3.18a.996.996 0 11-1.41 1.41zm-4.92 7.42L12 18.17l2.47-2.45a.996.996 0 011.41 0c.39.39.39 1.02 0 1.41l-3.17 3.17a.996.996 0 01-1.41 0l-3.17-3.18a.996.996 0 111.41-1.41z",
      fill: "currentColor"
    })
  });
}
function SvgExpandMoreIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 17",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M4 4.03l1.06 1.061 2.97-2.97L11 5.091l1.06-1.06L8.03 0 4 4.03zM12.06 12.091l-4.03 4.03L4 12.091l1.06-1.06L8.03 14 11 11.03l1.06 1.061z",
      fill: "currentColor"
    })
  });
}
function ExpandMoreIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgExpandMoreIconV2 : SvgExpandMoreIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgFileCodeIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M13.17 2c.53 0 1.04.21 1.42.59l4.82 4.83c.38.37.59.88.59 1.41V20c0 1.1-.9 2-2 2H5.99C4.89 22 4 21.1 4 20V4c0-1.1.9-2 2-2h7.17zm2.024 12.592l-1.9 1.9a.984.984 0 000 1.4c.39.39 1.01.39 1.4 0l2.59-2.6a.996.996 0 000-1.41l-2.59-2.59a.984.984 0 00-1.4 0 .984.984 0 000 1.4l1.9 1.9zm-4.487-1.9l-1.9 1.9 1.9 1.9a.984.984 0 010 1.4.984.984 0 01-1.4 0l-2.59-2.59a.996.996 0 010-1.41l2.59-2.6a.984.984 0 011.4 0 .984.984 0 010 1.4zM13 3.5V8c0 .55.45 1 1 1h4.5L13 3.5z",
      fill: "currentColor"
    })
  });
}
function SvgFileCodeIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 17",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M2 1.75A.75.75 0 012.75 1h6a.75.75 0 01.53.22l4.5 4.5c.141.14.22.331.22.53V10h-1.5V7H8.75A.75.75 0 018 6.25V2.5H3.5V16H2V1.75zm7.5 1.81l1.94 1.94H9.5V3.56z",
      fill: "currentColor"
    }), jsx("path", {
      d: "M7.47 9.97L4.44 13l3.03 3.03 1.06-1.06L6.56 13l1.97-1.97-1.06-1.06zM11.03 9.97l-1.06 1.06L11.94 13l-1.97 1.97 1.06 1.06L14.06 13l-3.03-3.03z",
      fill: "currentColor"
    })]
  });
}
function FileCodeIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgFileCodeIconV2 : SvgFileCodeIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgFileDocumentIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M13.17 2c.53 0 1.04.21 1.42.59l4.82 4.83c.38.37.59.88.59 1.41V20c0 1.1-.9 2-2 2H5.99C4.89 22 4 21.1 4 20V4c0-1.1.9-2 2-2h7.17zM9 18h6c.55 0 1-.45 1-1s-.45-1-1-1H9c-.55 0-1 .45-1 1s.45 1 1 1zm6-4H9c-.55 0-1-.45-1-1s.45-1 1-1h6c.55 0 1 .45 1 1s-.45 1-1 1zM13 3.5V8c0 .55.45 1 1 1h4.5L13 3.5z",
      fill: "currentColor"
    })
  });
}
function SvgFileDocumentIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M2 1.75A.75.75 0 012.75 1h6a.75.75 0 01.53.22l4.5 4.5c.141.14.22.331.22.53V10h-1.5V7H8.75A.75.75 0 018 6.25V2.5H3.5V16H2V1.75zm7.5 1.81l1.94 1.94H9.5V3.56z",
      fill: "currentColor"
    }), jsx("path", {
      d: "M5 11.5V13h9v-1.5H5zM14 16H5v-1.5h9V16z",
      fill: "currentColor"
    })]
  });
}
function FileDocumentIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgFileDocumentIconV2 : SvgFileDocumentIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgFileIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M13.17 2c.53 0 1.04.21 1.42.59l4.82 4.83c.38.37.59.88.59 1.41V20c0 1.1-.9 2-2 2H5.99C4.89 22 4 21.1 4 20V4c0-1.1.9-2 2-2h7.17zM13 3.5V8c0 .55.45 1 1 1h4.5L13 3.5z",
      fill: "currentColor"
    })
  });
}
function SvgFileIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M2 1.75A.75.75 0 012.75 1h6a.75.75 0 01.53.22l4.5 4.5c.141.14.22.331.22.53v9a.75.75 0 01-.75.75H2.75a.75.75 0 01-.75-.75V1.75zm1.5.75v12h9V7H8.75A.75.75 0 018 6.25V2.5H3.5zm6 1.06l1.94 1.94H9.5V3.56z",
      fill: "currentColor"
    })
  });
}
function FileIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgFileIconV2 : SvgFileIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgFileImageIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M14.59 2.59c-.38-.38-.89-.59-1.42-.59H6c-1.1 0-2 .9-2 2v16c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8.83c0-.53-.21-1.04-.59-1.41l-4.82-4.83zM16.605 18H7.383a.5.5 0 01-.429-.757l2.217-3.694a.5.5 0 01.782-.096l1.5 1.5a.05.05 0 00.08-.01l1.549-2.711a.5.5 0 01.86-.014l3.089 5.02a.5.5 0 01-.426.762zM13 3.5V8c0 .55.45 1 1 1h4.5L13 3.5z",
      fill: "currentColor"
    })
  });
}
function SvgFileImageIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M2 1.75A.75.75 0 012.75 1h6a.75.75 0 01.53.22l4.5 4.5c.141.14.22.331.22.53V10h-1.5V7H8.75A.75.75 0 018 6.25V2.5H3.5V16H2V1.75zm7.5 1.81l1.94 1.94H9.5V3.56z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M10.466 10a.75.75 0 00-.542.27l-3.75 4.5A.75.75 0 006.75 16h6.5a.75.75 0 00.75-.75V13.5a.75.75 0 00-.22-.53l-2.75-2.75a.75.75 0 00-.564-.22zm2.034 3.81v.69H8.351l2.2-2.639 1.949 1.95zM6.5 7.25a2.25 2.25 0 100 4.5 2.25 2.25 0 000-4.5zM5.75 9.5a.75.75 0 111.5 0 .75.75 0 01-1.5 0z",
      fill: "currentColor"
    })]
  });
}
function FileImageIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgFileImageIconV2 : SvgFileImageIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgFileModelIconV1(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M16 17a1 1 0 11-2 0 1 1 0 012 0zM9 12a1 1 0 100-2 1 1 0 000 2z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M14.59 2.59c-.38-.38-.89-.59-1.42-.59H6c-1.1 0-2 .9-2 2v16c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8.83c0-.53-.21-1.04-.59-1.41l-4.82-4.83zM13 8V3.5L18.5 9H14c-.55 0-1-.45-1-1zm2 12a3 3 0 10-1.293-5.708l-2-1.999a3 3 0 10-1.414 1.414l2 2A3 3 0 0015 20z",
      fill: "currentColor"
    })]
  });
}
function SvgFileModelIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M2.75 1a.75.75 0 00-.75.75V16h1.5V2.5H8v3.75c0 .414.336.75.75.75h3.75v3H14V6.25a.75.75 0 00-.22-.53l-4.5-4.5A.75.75 0 008.75 1h-6zm8.69 4.5L9.5 3.56V5.5h1.94z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M11.75 11.5a2.25 2.25 0 11-2.03 1.28l-.5-.5a2.25 2.25 0 111.06-1.06l.5.5c.294-.141.623-.22.97-.22zm.75 2.25a.75.75 0 10-1.5 0 .75.75 0 001.5 0zM8.25 9.5a.75.75 0 110 1.5.75.75 0 010-1.5z",
      fill: "currentColor"
    })]
  });
}
function FileModelIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgFileModelIconV2 : SvgFileModelIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgFilterIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M3 7c0 .55.45 1 1 1h16c.55 0 1-.45 1-1s-.45-1-1-1H4c-.55 0-1 .45-1 1zm8 11h2c.55 0 1-.45 1-1s-.45-1-1-1h-2c-.55 0-1 .45-1 1s.45 1 1 1zm6-5H7c-.55 0-1-.45-1-1s.45-1 1-1h10c.55 0 1 .45 1 1s-.45 1-1 1z",
      fill: "currentColor"
    })
  });
}
function SvgFilterIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M1 1.75A.75.75 0 011.75 1h12.5a.75.75 0 01.75.75V4a.75.75 0 01-.22.53L10 9.31v4.94a.75.75 0 01-.75.75h-2.5a.75.75 0 01-.75-.75V9.31L1.22 4.53A.75.75 0 011 4V1.75zm1.5.75v1.19l4.78 4.78c.141.14.22.331.22.53v4.5h1V9a.75.75 0 01.22-.53l4.78-4.78V2.5h-11z",
      fill: "currentColor"
    })
  });
}
function FilterIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgFilterIconV2 : SvgFilterIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgFolderFillIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M10.59 4.59C10.21 4.21 9.7 4 9.17 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-1.41-1.41z",
      fill: "currentColor"
    })
  });
}
function SvgFolderFillIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M.75 2a.75.75 0 00-.75.75v10.5c0 .414.336.75.75.75h14.5a.75.75 0 00.75-.75v-8.5a.75.75 0 00-.75-.75H7.81L6.617 2.805A2.75 2.75 0 004.672 2H.75z",
      fill: "currentColor"
    })
  });
}
function FolderFillIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgFolderFillIconV2 : SvgFolderFillIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgFolderIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M2 5a3 3 0 013-3h3.93a3 3 0 012.496 1.336L12.536 5H19a3 3 0 013 3v11a3 3 0 01-3 3H5a3 3 0 01-3-3V5zm3-1a1 1 0 00-1 1v14a1 1 0 001 1h14a1 1 0 001-1V8a1 1 0 00-1-1h-7.535L9.762 4.445A1 1 0 008.93 4H5z",
      fill: "currentColor"
    })
  });
}
function SvgFolderIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M0 2.75A.75.75 0 01.75 2h3.922c.729 0 1.428.29 1.944.805L7.811 4h7.439a.75.75 0 01.75.75v8.5a.75.75 0 01-.75.75H.75a.75.75 0 01-.75-.75V2.75zm1.5.75v9h13v-7h-7a.75.75 0 01-.53-.22L5.555 3.866a1.25 1.25 0 00-.883-.366H1.5z",
      fill: "currentColor"
    })
  });
}
function FolderIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgFolderIconV2 : SvgFolderIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgForkIconV1(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M8 6a1 1 0 100-2 1 1 0 000 2zm0 2a3 3 0 100-6 3 3 0 000 6zM8 20a1 1 0 100-2 1 1 0 000 2zm0 2a3 3 0 100-6 3 3 0 000 6zM16 20a1 1 0 100-2 1 1 0 000 2zm0 2a3 3 0 100-6 3 3 0 000 6z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M7 17V7h2v10H7zM12 12H9v-2h3a5 5 0 015 5v3h-2v-3a3 3 0 00-3-3z",
      fill: "currentColor"
    })]
  });
}
function SvgForkIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M2 2.75a2.75 2.75 0 113.5 2.646V6.75h3.75A2.75 2.75 0 0112 9.5v.104a2.751 2.751 0 11-1.5 0V9.5c0-.69-.56-1.25-1.25-1.25H5.5v1.354a2.751 2.751 0 11-1.5 0V5.396A2.751 2.751 0 012 2.75zM4.75 1.5a1.25 1.25 0 100 2.5 1.25 1.25 0 000-2.5zM3.5 12.25a1.25 1.25 0 112.5 0 1.25 1.25 0 01-2.5 0zm6.5 0a1.25 1.25 0 112.5 0 1.25 1.25 0 01-2.5 0z",
      fill: "currentColor"
    })
  });
}
function ForkIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgForkIconV2 : SvgForkIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgFullscreenExitIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M6 8h2V6c0-.55.45-1 1-1s1 .45 1 1v3c0 .55-.45 1-1 1H6c-.55 0-1-.45-1-1s.45-1 1-1zm2 8H6c-.55 0-1-.45-1-1s.45-1 1-1h3c.55 0 1 .45 1 1v3c0 .55-.45 1-1 1s-1-.45-1-1v-2zm7 3c.55 0 1-.45 1-1v-2h2c.55 0 1-.45 1-1s-.45-1-1-1h-3c-.55 0-1 .45-1 1v3c0 .55.45 1 1 1zm1-13v2h2c.55 0 1 .45 1 1s-.45 1-1 1h-3c-.55 0-1-.45-1-1V6c0-.55.45-1 1-1s1 .45 1 1z",
      fill: "currentColor"
    })
  });
}
function SvgFullscreenExitIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M6 1v4.25a.75.75 0 01-.75.75H1V4.5h3.5V1H6zM10 15v-4.25a.75.75 0 01.75-.75H15v1.5h-3.5V15H10zM10.75 6H15V4.5h-3.5V1H10v4.25c0 .414.336.75.75.75zM1 10h4.25a.75.75 0 01.75.75V15H4.5v-3.5H1V10z",
      fill: "currentColor"
    })
  });
}
function FullscreenExitIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgFullscreenExitIconV2 : SvgFullscreenExitIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgFullscreenIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M7 9c0 .55-.45 1-1 1s-1-.45-1-1V6c0-.55.45-1 1-1h3c.55 0 1 .45 1 1s-.45 1-1 1H7v2zm-2 6c0-.55.45-1 1-1s1 .45 1 1v2h2c.55 0 1 .45 1 1s-.45 1-1 1H6c-.55 0-1-.45-1-1v-3zm12 2h-2c-.55 0-1 .45-1 1s.45 1 1 1h3c.55 0 1-.45 1-1v-3c0-.55-.45-1-1-1s-1 .45-1 1v2zM15 7c-.55 0-1-.45-1-1s.45-1 1-1h3c.55 0 1 .45 1 1v3c0 .55-.45 1-1 1s-1-.45-1-1V7h-2z",
      fill: "currentColor"
    })
  });
}
function SvgFullscreenIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M6 1H1.75a.75.75 0 00-.75.75V6h1.5V2.5H6V1zM10 2.5V1h4.25a.75.75 0 01.75.75V6h-1.5V2.5H10zM10 13.5h3.5V10H15v4.25a.75.75 0 01-.75.75H10v-1.5zM2.5 10v3.5H6V15H1.75a.75.75 0 01-.75-.75V10h1.5z",
      fill: "currentColor"
    })
  });
}
function FullscreenIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgFullscreenIconV2 : SvgFullscreenIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgGearFillIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M19.502 12c0 .34-.03.66-.07.98l2.11 1.65c.19.15.24.42.12.64l-2 3.46c-.12.22-.38.31-.61.22l-2.49-1c-.52.39-1.08.73-1.69.98l-.38 2.65c-.03.24-.24.42-.49.42h-4c-.25 0-.46-.18-.49-.42l-.38-2.65c-.61-.25-1.17-.58-1.69-.98l-2.49 1c-.22.08-.49 0-.61-.22l-2-3.46a.505.505 0 01.12-.64l2.11-1.65a7.93 7.93 0 01-.07-.98c0-.33.03-.66.07-.98l-2.11-1.65a.493.493 0 01-.12-.64l2-3.46c.12-.22.38-.31.61-.22l2.49 1c.52-.39 1.08-.73 1.69-.98l.38-2.65c.03-.24.24-.42.49-.42h4c.25 0 .46.18.49.42l.38 2.65c.61.25 1.17.58 1.69.98l2.49-1c.22-.08.49 0 .61.22l2 3.46c.12.22.07.49-.12.64l-2.11 1.65c.04.32.07.64.07.98zm-11 0c0 1.93 1.57 3.5 3.5 3.5s3.5-1.57 3.5-3.5-1.57-3.5-3.5-3.5-3.5 1.57-3.5 3.5z",
      fill: "currentColor"
    })
  });
}
function SvgGearFillIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M7.965 0c-.34 0-.675.021-1.004.063a.75.75 0 00-.62.51l-.639 1.946c-.21.087-.413.185-.61.294L3.172 2.1a.75.75 0 00-.784.165c-.481.468-.903.996-1.255 1.572a.75.75 0 00.013.802l1.123 1.713a5.898 5.898 0 00-.15.66L.363 8.07a.75.75 0 00-.36.716c.067.682.22 1.34.447 1.962a.75.75 0 00.635.489l2.042.19c.13.184.271.36.422.529l-.27 2.032a.75.75 0 00.336.728 7.97 7.97 0 001.812.874.75.75 0 00.778-.192l1.422-1.478a5.924 5.924 0 00.677 0l1.422 1.478a.75.75 0 00.778.192 7.972 7.972 0 001.812-.874.75.75 0 00.335-.728l-.269-2.032a5.94 5.94 0 00.422-.529l2.043-.19a.75.75 0 00.634-.49c.228-.621.38-1.279.447-1.961a.75.75 0 00-.36-.716l-1.756-1.056a5.89 5.89 0 00-.15-.661l1.123-1.713a.75.75 0 00.013-.802 8.034 8.034 0 00-1.255-1.572.75.75 0 00-.784-.165l-1.92.713c-.197-.109-.4-.207-.61-.294L9.589.573a.75.75 0 00-.619-.51A8.07 8.07 0 007.965 0zm.02 10.25a2.25 2.25 0 100-4.5 2.25 2.25 0 000 4.5z",
      fill: "currentColor"
    })
  });
}
function GearFillIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgGearFillIconV2 : SvgGearFillIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgGearIconV1(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M14.616 2l.68 2.042 1.926-.962 3.698 3.698-.962 1.926 2.042.68v5.232l-2.042.68.963 1.926-3.7 3.698-1.925-.962-.68 2.042H9.384l-.68-2.042-1.926.963-3.698-3.7.962-1.925L2 14.616V9.384l2.042-.68-.962-1.926L6.777 3.08l1.926.962L9.384 2h5.232zm-3.79 2l-.954 2.862-2.699-1.349-1.66 1.66 1.35 2.699L4 10.826v2.348l2.862.954-1.349 2.699 1.66 1.66 2.699-1.35.954 2.863h2.348l.954-2.862 2.699 1.349 1.66-1.66-1.35-2.699L20 13.174v-2.348l-2.862-.954 1.349-2.699-1.66-1.66-2.699 1.35L13.174 4h-2.348z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M12 14a2 2 0 100-4 2 2 0 000 4zm0 2a4 4 0 100-8 4 4 0 000 8z",
      fill: "currentColor"
    })]
  });
}
function SvgGearIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsxs("g", {
      clipPath: "url(#GearIcon_svg__clip0_13123_35019)",
      fillRule: "evenodd",
      clipRule: "evenodd",
      fill: "currentColor",
      children: [jsx("path", {
        d: "M7.984 5a3 3 0 100 6 3 3 0 000-6zm-1.5 3a1.5 1.5 0 113 0 1.5 1.5 0 01-3 0z"
      }), jsx("path", {
        d: "M7.965 0c-.34 0-.675.021-1.004.063a.75.75 0 00-.62.51l-.639 1.946c-.21.087-.413.185-.61.294L3.172 2.1a.75.75 0 00-.784.165c-.481.468-.903.996-1.255 1.572a.75.75 0 00.013.802l1.123 1.713a5.898 5.898 0 00-.15.66L.363 8.07a.75.75 0 00-.36.716c.067.682.22 1.34.447 1.962a.75.75 0 00.635.489l2.042.19c.13.184.271.36.422.529l-.27 2.032a.75.75 0 00.336.728 7.97 7.97 0 001.812.874.75.75 0 00.778-.192l1.422-1.478a5.924 5.924 0 00.677 0l1.422 1.478a.75.75 0 00.778.192 7.972 7.972 0 001.812-.874.75.75 0 00.335-.728l-.269-2.032a5.94 5.94 0 00.422-.529l2.043-.19a.75.75 0 00.634-.49c.228-.621.38-1.279.447-1.961a.75.75 0 00-.36-.716l-1.756-1.056a5.89 5.89 0 00-.15-.661l1.123-1.713a.75.75 0 00.013-.802 8.034 8.034 0 00-1.255-1.572.75.75 0 00-.784-.165l-1.92.713c-.197-.109-.4-.207-.61-.294L9.589.573a.75.75 0 00-.619-.51A8.071 8.071 0 007.965 0zm-.95 3.328l.598-1.819a6.62 6.62 0 01.705 0l.597 1.819a.75.75 0 00.472.476c.345.117.67.275.97.468a.75.75 0 00.668.073l1.795-.668c.156.176.303.36.44.552l-1.05 1.6a.75.75 0 00-.078.667c.12.333.202.685.24 1.05a.75.75 0 00.359.567l1.642.988c-.04.234-.092.463-.156.687l-1.909.178a.75.75 0 00-.569.353c-.19.308-.416.59-.672.843a.75.75 0 00-.219.633l.252 1.901a6.48 6.48 0 01-.635.306l-1.33-1.381a.75.75 0 00-.63-.225 4.483 4.483 0 01-1.08 0 .75.75 0 00-.63.225l-1.33 1.381a6.473 6.473 0 01-.634-.306l.252-1.9a.75.75 0 00-.219-.634 4.449 4.449 0 01-.672-.843.75.75 0 00-.569-.353l-1.909-.178a6.456 6.456 0 01-.156-.687L3.2 8.113a.75.75 0 00.36-.567c.037-.365.118-.717.239-1.05a.75.75 0 00-.078-.666L2.67 4.229c.137-.192.284-.376.44-.552l1.795.668a.75.75 0 00.667-.073c.3-.193.626-.351.97-.468a.75.75 0 00.472-.476z"
      })]
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "GearIcon_svg__clip0_13123_35019",
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
function GearIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgGearIconV2 : SvgGearIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgGridDashIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M7 5h2V3H7v2zm0 8h2v-2H7v2zm2 8H7v-2h2v2zm2-4h2v-2h-2v2zm2 4h-2v-2h2v2zM3 21h2v-2H3v2zm2-4H3v-2h2v2zm-2-4h2v-2H3v2zm2-4H3V7h2v2zM3 5h2V3H3v2zm10 8h-2v-2h2v2zm6 4h2v-2h-2v2zm2-4h-2v-2h2v2zm-2 8h2v-2h-2v2zm2-12h-2V7h2v2zM11 9h2V7h-2v2zm8-4V3h2v2h-2zm-8 0h2V3h-2v2zm6 16h-2v-2h2v2zm-2-8h2v-2h-2v2zm2-8h-2V3h2v2z",
      fill: "currentColor"
    })
  });
}
function SvgGridDashIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M1 1.75V4h1.5V2.5H4V1H1.75a.75.75 0 00-.75.75zM15 14.25V12h-1.5v1.5H12V15h2.25a.75.75 0 00.75-.75zM12 1h2.25a.75.75 0 01.75.75V4h-1.5V2.5H12V1zM1.75 15H4v-1.5H2.5V12H1v2.25a.75.75 0 00.75.75zM10 2.5H6V1h4v1.5zM6 15h4v-1.5H6V15zM13.5 10V6H15v4h-1.5zM1 6v4h1.5V6H1z",
      fill: "currentColor"
    })
  });
}
function GridDashIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgGridDashIconV2 : SvgGridDashIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgGridIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M5 6a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1H6a1 1 0 01-1-1V6zM5 14a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1H6a1 1 0 01-1-1v-4zM13 6a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1V6zM13 14a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z",
      fill: "currentColor"
    })
  });
}
function SvgGridIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M1.75 1a.75.75 0 00-.75.75v4.5c0 .414.336.75.75.75h4.5A.75.75 0 007 6.25v-4.5A.75.75 0 006.25 1h-4.5zm.75 4.5v-3h3v3h-3zM1.75 9a.75.75 0 00-.75.75v4.5c0 .414.336.75.75.75h4.5a.75.75 0 00.75-.75v-4.5A.75.75 0 006.25 9h-4.5zm.75 4.5v-3h3v3h-3zM9 1.75A.75.75 0 019.75 1h4.5a.75.75 0 01.75.75v4.49a.75.75 0 01-.75.75h-4.5A.75.75 0 019 6.24V1.75zm1.5.75v2.99h3V2.5h-3zM9.75 9a.75.75 0 00-.75.75v4.5c0 .414.336.75.75.75h4.5a.75.75 0 00.75-.75v-4.5a.75.75 0 00-.75-.75h-4.5zm.75 4.5v-3h3v3h-3z",
      fill: "currentColor"
    })
  });
}
function GridIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgGridIconV2 : SvgGridIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgH1IconV1(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M7 7a1 1 0 011 1v3h3V8a1 1 0 112 0v8a1 1 0 11-2 0v-3H8v3a1 1 0 11-2 0V8a1 1 0 011-1zM15 12a1 1 0 011-1h.5a1.5 1.5 0 011.5 1.5V16a1 1 0 11-2 0v-3a1 1 0 01-1-1z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M4 1a3 3 0 00-3 3v16a3 3 0 003 3h16a3 3 0 003-3V4a3 3 0 00-3-3H4zM3 4a1 1 0 011-1h16a1 1 0 011 1v16a1 1 0 01-1 1H4a1 1 0 01-1-1V4z",
      fill: "currentColor"
    })]
  });
}
function SvgH1IconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M1 3v10h1.5V8.75H6V13h1.5V3H6v4.25H2.5V3H1zM11.25 3A2.25 2.25 0 019 5.25v1.5c.844 0 1.623-.279 2.25-.75v5.5H9V13h6v-1.5h-2.25V3h-1.5z",
      fill: "currentColor"
    })
  });
}
function H1Icon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgH1IconV2 : SvgH1IconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgH2IconV1(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M6 7a1 1 0 011 1v3h3V8a1 1 0 112 0v8a1 1 0 11-2 0v-3H7v3a1 1 0 11-2 0V8a1 1 0 011-1zM14 10a1 1 0 011-1h2.5c.83 0 1.5.673 1.5 1.5v2a1.5 1.5 0 01-1.5 1.5H16v1h2a1 1 0 110 2h-2.5a1.5 1.5 0 01-1.5-1.5v-2a1.5 1.5 0 011.5-1.5H17v-1h-2a1 1 0 01-1-1z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M4 1a3 3 0 00-3 3v16a3 3 0 003 3h16a3 3 0 003-3V4a3 3 0 00-3-3H4zM3 4a1 1 0 011-1h16a1 1 0 011 1v16a1 1 0 01-1 1H4a1 1 0 01-1-1V4z",
      fill: "currentColor"
    })]
  });
}
function SvgH2IconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M1 3v10h1.5V8.75H6V13h1.5V3H6v4.25H2.5V3H1zM11.75 3A2.75 2.75 0 009 5.75V6h1.5v-.25c0-.69.56-1.25 1.25-1.25h.39a1.36 1.36 0 01.746 2.498L10.692 8.44A3.75 3.75 0 009 11.574V13h6v-1.5h-4.499a2.25 2.25 0 011.014-1.807l2.194-1.44A2.86 2.86 0 0012.14 3h-.389z",
      fill: "currentColor"
    })
  });
}
function H2Icon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgH2IconV2 : SvgH2IconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgH3IconV1(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M6 7a1 1 0 011 1v3h3V8a1 1 0 112 0v8a1 1 0 11-2 0v-3H7v3a1 1 0 11-2 0V8a1 1 0 011-1zM17.5 17a1.5 1.5 0 001.5-1.5v-7A1.5 1.5 0 0017.5 7H15a1 1 0 100 2h2v2h-2a1 1 0 100 2h2v2h-2a1 1 0 100 2h2.5z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M1 4a3 3 0 013-3h16a3 3 0 013 3v16a3 3 0 01-3 3H4a3 3 0 01-3-3V4zm3-1a1 1 0 00-1 1v16a1 1 0 001 1h16a1 1 0 001-1V4a1 1 0 00-1-1H4z",
      fill: "currentColor"
    })]
  });
}
function SvgH3IconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M1 3h1.5v4.25H6V3h1.5v10H6V8.75H2.5V13H1V3zM9 5.75A2.75 2.75 0 0111.75 3h.375a2.875 2.875 0 011.937 5 2.875 2.875 0 01-1.937 5h-.375A2.75 2.75 0 019 10.25V10h1.5v.25c0 .69.56 1.25 1.25 1.25h.375a1.375 1.375 0 100-2.75H11v-1.5h1.125a1.375 1.375 0 100-2.75h-.375c-.69 0-1.25.56-1.25 1.25V6H9v-.25z",
      fill: "currentColor"
    })
  });
}
function H3Icon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgH3IconV2 : SvgH3IconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgHistoryIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M4.144 12.002c0-5.05 4.17-9.14 9.26-9 4.69.13 8.61 4.05 8.74 8.74.14 5.09-3.95 9.26-9 9.26-2.09 0-4-.71-5.52-1.91-.47-.36-.5-1.07-.08-1.49.36-.36.92-.39 1.32-.08 1.18.93 2.67 1.48 4.28 1.48 3.9 0 7.05-3.19 7-7.1-.05-3.72-3.18-6.85-6.9-6.9-3.92-.05-7.1 3.1-7.1 7h1.79a.5.5 0 01.36.85l-2.79 2.8c-.2.2-.51.2-.71 0l-2.79-2.8c-.32-.31-.1-.85.35-.85h1.79zm8-3.25c0-.41.34-.75.75-.75s.75.34.75.74v3.4l2.88 1.71c.35.21.47.67.26 1.03-.21.35-.67.47-1.03.26l-3.12-1.85c-.3-.18-.49-.51-.49-.86v-3.68z",
      fill: "currentColor"
    })
  });
}
function SvgHistoryIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsxs("g", {
      clipPath: "url(#HistoryIcon_svg__clip0_13123_35203)",
      fill: "currentColor",
      children: [jsx("path", {
        d: "M3.507 7.73l.963-.962 1.06 1.06-2.732 2.732L-.03 7.732l1.06-1.06.979.978a7 7 0 112.041 5.3l1.061-1.06a5.5 5.5 0 10-1.604-4.158z"
      }), jsx("path", {
        d: "M8.25 8V4h1.5v3.69l1.78 1.78-1.06 1.06-2-2A.75.75 0 018.25 8z"
      })]
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "HistoryIcon_svg__clip0_13123_35203",
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
function HistoryIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgHistoryIconV2 : SvgHistoryIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgHomeIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M9.998 19.328v-5h4v5c0 .55.45 1 1 1h3c.55 0 1-.45 1-1v-7h1.7c.46 0 .68-.57.33-.87l-8.36-7.53c-.38-.34-.96-.34-1.34 0l-8.36 7.53c-.34.3-.13.87.33.87h1.7v7c0 .55.45 1 1 1h3c.55 0 1-.45 1-1z",
      fill: "currentColor"
    })
  });
}
function SvgHomeIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M7.625 1.1a.75.75 0 01.75 0l6.25 3.61a.75.75 0 01.375.65v8.89a.75.75 0 01-.75.75h-4.5a.75.75 0 01-.75-.75V10H7v4.25a.75.75 0 01-.75.75h-4.5a.75.75 0 01-.75-.75V5.355a.75.75 0 01.375-.65L7.625 1.1zM2.5 5.79V13.5h3V9.25a.75.75 0 01.75-.75h3.5a.75.75 0 01.75.75v4.25h3V5.792L8 2.616 2.5 5.789z",
      fill: "currentColor"
    })
  });
}
function HomeIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgHomeIconV2 : SvgHomeIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgImageIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M21 5v14c0 1.1-.9 2-2 2H5c-1.1 0-2-.9-2-2V5c0-1.1.9-2 2-2h14c1.1 0 2 .9 2 2zM11 16.51l-2.1-2.53a.493.493 0 00-.78.02l-2.49 3.2c-.26.33-.03.81.39.81h11.99a.5.5 0 00.4-.8l-3.51-4.68c-.2-.27-.6-.27-.8-.01L11 16.51z",
      fill: "currentColor"
    })
  });
}
function SvgImageIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M6.25 3.998a2.25 2.25 0 100 4.5 2.25 2.25 0 000-4.5zm-.75 2.25a.75.75 0 111.5 0 .75.75 0 01-1.5 0z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M1 1.75A.75.75 0 011.75 1h12.5a.75.75 0 01.75.75v12.492a.75.75 0 01-.75.75H5.038l-.009.009-.008-.009H1.75a.75.75 0 01-.75-.75V1.75zm12.5 11.742H6.544l4.455-4.436 2.47 2.469.031-.03v1.997zm0-10.992v6.934l-1.97-1.968a.75.75 0 00-1.06-.001l-6.052 6.027H2.5V2.5h11z",
      fill: "currentColor"
    })]
  });
}
function ImageIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgImageIconV2 : SvgImageIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgIndentDecreaseIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M4 5c-.55 0-1-.45-1-1s.45-1 1-1h16c.55 0 1 .45 1 1s-.45 1-1 1H4zm2.14 10.14l-2.79-2.79a.492.492 0 01.01-.7l2.79-2.79c.31-.32.85-.1.85.35v5.58c0 .45-.54.67-.86.35zM20 17h-8c-.55 0-1-.45-1-1s.45-1 1-1h8c.55 0 1 .45 1 1s-.45 1-1 1zm0 4c.55 0 1-.45 1-1s-.45-1-1-1H4c-.55 0-1 .45-1 1s.45 1 1 1h16zM12 9h8c.55 0 1-.45 1-1s-.45-1-1-1h-8c-.55 0-1 .45-1 1s.45 1 1 1zm8 4h-8c-.55 0-1-.45-1-1s.45-1 1-1h8c.55 0 1 .45 1 1s-.45 1-1 1z",
      fill: "currentColor"
    })
  });
}
function SvgIndentDecreaseIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M16 2H0v1.5h16V2zM16 5.5H7V7h9V5.5zM16 9H7v1.5h9V9zM16 12.5H0V14h16v-1.5zM3.97 11.03L.94 8l3.03-3.03 1.06 1.06L3.06 8l1.97 1.97-1.06 1.06z",
      fill: "currentColor"
    })
  });
}
function IndentDecreaseIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgIndentDecreaseIconV2 : SvgIndentDecreaseIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgIndentIncreaseIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M4 5c-.55 0-1-.45-1-1s.45-1 1-1h16c.55 0 1 .45 1 1s-.45 1-1 1H4zm-1 9.8V9.21a.5.5 0 01.85-.36l2.79 2.8c.2.2.2.51 0 .71l-2.79 2.79c-.31.32-.85.1-.85-.35zM4 21c-.55 0-1-.45-1-1s.45-1 1-1h16c.55 0 1 .45 1 1s-.45 1-1 1H4zm8-4h8c.55 0 1-.45 1-1s-.45-1-1-1h-8c-.55 0-1 .45-1 1s.45 1 1 1zm0-8h8c.55 0 1-.45 1-1s-.45-1-1-1h-8c-.55 0-1 .45-1 1s.45 1 1 1zm8 4h-8c-.55 0-1-.45-1-1s.45-1 1-1h8c.55 0 1 .45 1 1s-.45 1-1 1z",
      fill: "currentColor"
    })
  });
}
function SvgIndentIncreaseIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M16 2H0v1.5h16V2zM16 5.5H7V7h9V5.5zM16 9H7v1.5h9V9zM16 12.5H0V14h16v-1.5zM2.03 4.97L5.06 8l-3.03 3.03L.97 9.97 2.94 8 .97 6.03l1.06-1.06z",
      fill: "currentColor"
    })
  });
}
function IndentIncreaseIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgIndentIncreaseIconV2 : SvgIndentIncreaseIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgInfinityIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M8 6.94l1.591-1.592a3.75 3.75 0 110 5.304L8 9.06l-1.591 1.59a3.75 3.75 0 110-5.303L8 6.94zm2.652-.531a2.25 2.25 0 110 3.182L9.06 8l1.59-1.591zM6.939 8L5.35 6.409a2.25 2.25 0 100 3.182l1.588-1.589L6.939 8z",
      fill: "currentColor"
    })
  });
}
function SvgInfinityIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M8 6.94l1.59-1.592a3.75 3.75 0 110 5.304L8 9.06l-1.591 1.59a3.75 3.75 0 110-5.303L8 6.94zm2.652-.531a2.25 2.25 0 110 3.182L9.06 8l1.59-1.591zM6.939 8L5.35 6.409a2.25 2.25 0 100 3.182l1.588-1.589L6.939 8z",
      fill: "currentColor"
    })
  });
}
function InfinityIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgInfinityIconV2 : SvgInfinityIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgInfoFillIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 15c-.55 0-1-.45-1-1v-4c0-.55.45-1 1-1s1 .45 1 1v4c0 .55-.45 1-1 1zm-1-8h2V7h-2v2z",
      fill: "currentColor"
    })
  });
}
function SvgInfoFillIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M16 8A8 8 0 110 8a8 8 0 0116 0zm-8.75 3V7h1.5v4h-1.5zM8 4.5A.75.75 0 118 6a.75.75 0 010-1.5z",
      fill: "currentColor"
    })
  });
}
function InfoFillIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgInfoFillIconV2 : SvgInfoFillIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgInfoIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 5v2h2V7h-2zm2 9c0 .55-.45 1-1 1s-1-.45-1-1v-4c0-.55.45-1 1-1s1 .45 1 1v4zm-9-4c0 4.41 3.59 8 8 8s8-3.59 8-8-3.59-8-8-8-8 3.59-8 8z",
      fill: "currentColor"
    })
  });
}
function SvgInfoIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M7.25 11V7h1.5v4h-1.5zM8 4.5A.75.75 0 118 6a.75.75 0 010-1.5z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M0 8a8 8 0 1116 0A8 8 0 010 8zm8-6.5a6.5 6.5 0 100 13 6.5 6.5 0 000-13z",
      fill: "currentColor"
    })]
  });
}
function InfoIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgInfoIconV2 : SvgInfoIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgKeyboardIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M4 5h16c1.1 0 2 .9 2 2v10c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2l.01-10c0-1.1.89-2 1.99-2zm9 3h-2v2h2V8zm-2 3h2v2h-2v-2zm-1-3H8v2h2V8zm-2 3h2v2H8v-2zm-3 2h2v-2H5v2zm2-3H5V8h2v2zm2 7h6c.55 0 1-.45 1-1s-.45-1-1-1H9c-.55 0-1 .45-1 1s.45 1 1 1zm7-4h-2v-2h2v2zm-2-3h2V8h-2v2zm5 3h-2v-2h2v2zm-2-3h2V8h-2v2z",
      fill: "currentColor"
    })
  });
}
function SvgKeyboardIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M.75 2a.75.75 0 00-.75.75v10.5c0 .414.336.75.75.75h14.5a.75.75 0 00.75-.75V2.75a.75.75 0 00-.75-.75H.75zm.75 10.5v-9h13v9h-13zm2.75-8h-1.5V6h1.5V4.5zm1.5 0V6h1.5V4.5h-1.5zm3 0V6h1.5V4.5h-1.5zm3 0V6h1.5V4.5h-1.5zm-1.5 2.75h-1.5v1.5h1.5v-1.5zm1.5 1.5v-1.5h1.5v1.5h-1.5zm-4.5 0v-1.5h-1.5v1.5h1.5zm-3 0v-1.5h-1.5v1.5h1.5zM11 10H5v1.5h6V10z",
      fill: "currentColor"
    })
  });
}
function KeyboardIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgKeyboardIconV2 : SvgKeyboardIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgLayerIconV1(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M10 3a1 1 0 011-1h8.5A2.5 2.5 0 0122 4.5V13a1 1 0 11-2 0V4.5a.5.5 0 00-.5-.5H11a1 1 0 01-1-1z",
      fill: "currentColor"
    }), jsx("path", {
      d: "M6 7a1 1 0 011-1h8.5A2.5 2.5 0 0118 8.5V17a1 1 0 11-2 0V8.5a.5.5 0 00-.5-.5H7a1 1 0 01-1-1z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M2 12a2 2 0 012-2h8a2 2 0 012 2v8a2 2 0 01-2 2H4a2 2 0 01-2-2v-8zm2 0h8v8H4v-8z",
      fill: "currentColor"
    })]
  });
}
function SvgLayerIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M13.5 2.5H7V1h7.25a.75.75 0 01.75.75V9h-1.5V2.5z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M1 7.75A.75.75 0 011.75 7h6.5a.75.75 0 01.75.75v6.5a.75.75 0 01-.75.75h-6.5a.75.75 0 01-.75-.75v-6.5zm1.5.75v5h5v-5h-5z",
      fill: "currentColor"
    }), jsx("path", {
      d: "M4 5.32h6.5V12H12V4.57a.75.75 0 00-.75-.75H4v1.5z",
      fill: "currentColor"
    })]
  });
}
function LayerIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgLayerIconV2 : SvgLayerIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgLightningIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M7.37 3.675v9c0 .55.45 1 1 1h2v7.15c0 .51.67.69.93.25l5.19-8.9a.995.995 0 00-.86-1.5h-2.26l2.49-6.65a.994.994 0 00-.93-1.35H8.37c-.55 0-1 .45-1 1z",
      fill: "currentColor"
    })
  });
}
function SvgLightningIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M9.49.04a.75.75 0 01.51.71V6h3.25a.75.75 0 01.596 1.206l-6.5 8.5A.75.75 0 016 15.25V10H2.75a.75.75 0 01-.596-1.206l6.5-8.5A.75.75 0 019.491.04zM4.269 8.5H6.75a.75.75 0 01.75.75v3.785L11.732 7.5H9.25a.75.75 0 01-.75-.75V2.965L4.268 8.5z",
      fill: "currentColor"
    })
  });
}
function LightningIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgLightningIconV2 : SvgLightningIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgLinkIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M7 15h3c.55 0 1 .45 1 1s-.45 1-1 1H7c-2.76 0-5-2.24-5-5s2.24-5 5-5h3c.55 0 1 .45 1 1s-.45 1-1 1H7c-1.65 0-3 1.35-3 3s1.35 3 3 3zm10-8h-3c-.55 0-1 .45-1 1s.45 1 1 1h3c1.65 0 3 1.35 3 3s-1.35 3-3 3h-3c-.55 0-1 .45-1 1s.45 1 1 1h3c2.76 0 5-2.24 5-5s-2.24-5-5-5zm-9 5c0 .55.45 1 1 1h6c.55 0 1-.45 1-1s-.45-1-1-1H9c-.55 0-1 .45-1 1z",
      fill: "currentColor"
    })
  });
}
function SvgLinkIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M4 4h3v1.5H4a2.5 2.5 0 000 5h3V12H4a4 4 0 010-8zM12 10.5H9V12h3a4 4 0 000-8H9v1.5h3a2.5 2.5 0 010 5z",
      fill: "currentColor"
    }), jsx("path", {
      d: "M4 8.75h8v-1.5H4v1.5z",
      fill: "currentColor"
    })]
  });
}
function LinkIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgLinkIconV2 : SvgLinkIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgLinkOffIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M4.119 3.63a.996.996 0 10-1.41 1.41l2.4 2.4c-1.94.8-3.27 2.77-3.09 5.04.21 2.64 2.57 4.59 5.21 4.59h2.82c.52 0 .95-.43.95-.95s-.43-.95-.95-.95h-2.89c-1.63 0-3.1-1.19-3.25-2.82A3.095 3.095 0 016.659 9l2.1 2.1c-.43.09-.76.46-.76.92v.1c0 .52.43.95.95.95h1.78l2.27 2.27v1.73h1.73l3.3 3.3a.996.996 0 101.41-1.41L4.119 3.63zm17.82 7.67c-.37-2.47-2.62-4.23-5.12-4.23h-2.87c-.52 0-.95.43-.95.95s.43.95.95.95h2.9c1.6 0 3.04 1.14 3.22 2.73.17 1.43-.64 2.69-1.85 3.22l1.4 1.4c1.63-1.02 2.64-2.91 2.32-5.02zm-6.89-.23c.52 0 .95.43.95.95v.1c0 .16-.05.31-.12.44l-1.49-1.49h.66z",
      fill: "currentColor"
    })
  });
}
function SvgLinkOffIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M14.035 11.444A4 4 0 0012 4H9v1.5h3a2.5 2.5 0 01.917 4.826l1.118 1.118zM14 13.53L2.47 2l-1 1 1.22 1.22A4.002 4.002 0 004 12h3v-1.5H4a2.5 2.5 0 01-.03-5l1.75 1.75H4v1.5h3.22L13 14.53l1-1z",
      fill: "currentColor"
    }), jsx("path", {
      d: "M9.841 7.25l1.5 1.5H12v-1.5H9.841z",
      fill: "currentColor"
    })]
  });
}
function LinkOffIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgLinkOffIconV2 : SvgLinkOffIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgListBorderIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M20 3H4c-.55 0-1 .45-1 1v16c0 .55.45 1 1 1h16c.55 0 1-.45 1-1V4c0-.55-.45-1-1-1zM9 7H7v2h2V7zm7 2h-4c-.55 0-1-.45-1-1s.45-1 1-1h4c.55 0 1 .45 1 1s-.45 1-1 1zm0 4h-4c-.55 0-1-.45-1-1s.45-1 1-1h4c.55 0 1 .45 1 1s-.45 1-1 1zm-4 4h4c.55 0 1-.45 1-1s-.45-1-1-1h-4c-.55 0-1 .45-1 1s.45 1 1 1zm-5-6h2v2H7v-2zm2 4H7v2h2v-2zm-4 4h14V5H5v14z",
      fill: "currentColor"
    })
  });
}
function SvgListBorderIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M12 8.75H7v-1.5h5v1.5zM7 5.5h5V4H7v1.5zM12 12H7v-1.5h5V12zM4.75 5.5a.75.75 0 100-1.5.75.75 0 000 1.5zM5.5 8A.75.75 0 114 8a.75.75 0 011.5 0zM4.75 12a.75.75 0 100-1.5.75.75 0 000 1.5z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M1 1.75A.75.75 0 011.75 1h12.5a.75.75 0 01.75.75v12.5a.75.75 0 01-.75.75H1.75a.75.75 0 01-.75-.75V1.75zm1.5.75v11h11v-11h-11z",
      fill: "currentColor"
    })]
  });
}
function ListBorderIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgListBorderIconV2 : SvgListBorderIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgListIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M4 9c.55 0 1-.45 1-1s-.45-1-1-1-1 .45-1 1 .45 1 1 1zM5 12c0 .55-.45 1-1 1s-1-.45-1-1 .45-1 1-1 1 .45 1 1zM5 16c0 .55-.45 1-1 1s-1-.45-1-1 .45-1 1-1 1 .45 1 1zM20 13H8c-.55 0-1-.45-1-1s.45-1 1-1h12c.55 0 1 .45 1 1s-.45 1-1 1zM8 17h12c.55 0 1-.45 1-1s-.45-1-1-1H8c-.55 0-1 .45-1 1s.45 1 1 1zM8 9c-.55 0-1-.45-1-1s.45-1 1-1h12c.55 0 1 .45 1 1s-.45 1-1 1H8z",
      fill: "currentColor"
    })
  });
}
function SvgListIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M1.5 2.75a.75.75 0 11-1.5 0 .75.75 0 011.5 0zM3 2h13v1.5H3V2zM3 5.5h13V7H3V5.5zM3 9h13v1.5H3V9zM3 12.5h13V14H3v-1.5zM.75 7a.75.75 0 100-1.5.75.75 0 000 1.5zM1.5 13.25a.75.75 0 11-1.5 0 .75.75 0 011.5 0zM.75 10.5a.75.75 0 100-1.5.75.75 0 000 1.5z",
      fill: "currentColor"
    })
  });
}
function ListIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgListIconV2 : SvgListIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgLoadingIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M23.212 12a.788.788 0 01-.789-.788 9.57 9.57 0 00-.757-3.751 9.662 9.662 0 00-5.129-5.129 9.587 9.587 0 00-3.749-.755.788.788 0 010-1.577c1.513 0 2.983.296 4.365.882a11.128 11.128 0 013.562 2.403 11.157 11.157 0 013.283 7.927.785.785 0 01-.786.788z",
      fill: "currentColor"
    })
  });
}
function SvgLoadingIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M23.212 12a.788.788 0 01-.789-.788 9.57 9.57 0 00-.757-3.751 9.662 9.662 0 00-5.129-5.129 9.587 9.587 0 00-3.749-.755.788.788 0 010-1.577c1.513 0 2.983.296 4.365.882a11.128 11.128 0 013.562 2.403 11.157 11.157 0 013.283 7.927.785.785 0 01-.786.788z",
      fill: "currentColor"
    })
  });
}
function LoadingIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgLoadingIconV2 : SvgLoadingIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgLockFillIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M18 8.5h-1v-2c0-2.76-2.24-5-5-5s-5 2.24-5 5v2H6c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2v-10c0-1.1-.9-2-2-2zm-6 9c-1.1 0-2-.9-2-2s.9-2 2-2 2 .9 2 2-.9 2-2 2zm-3-11v2h6v-2c0-1.66-1.34-3-3-3s-3 1.34-3 3z",
      fill: "currentColor"
    })
  });
}
function SvgLockFillIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M12 6V4a4 4 0 00-8 0v2H2.75a.75.75 0 00-.75.75v8.5c0 .414.336.75.75.75h10.5a.75.75 0 00.75-.75v-8.5a.75.75 0 00-.75-.75H12zM5.5 6h5V4a2.5 2.5 0 00-5 0v2zm1.75 7V9h1.5v4h-1.5z",
      fill: "currentColor"
    })
  });
}
function LockFillIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgLockFillIconV2 : SvgLockFillIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgLockIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M17 8.5h1c1.1 0 2 .9 2 2v10c0 1.1-.9 2-2 2H6c-1.1 0-2-.9-2-2v-10c0-1.1.9-2 2-2h1v-2c0-2.76 2.24-5 5-5s5 2.24 5 5v2zm-5-5c-1.66 0-3 1.34-3 3v2h6v-2c0-1.66-1.34-3-3-3zm-5 17c-.55 0-1-.45-1-1v-8c0-.55.45-1 1-1h10c.55 0 1 .45 1 1v8c0 .55-.45 1-1 1H7zm7-5c0 1.1-.9 2-2 2s-2-.9-2-2 .9-2 2-2 2 .9 2 2z",
      fill: "currentColor"
    })
  });
}
function SvgLockIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M7.25 9v4h1.5V9h-1.5z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M12 6V4a4 4 0 00-8 0v2H2.75a.75.75 0 00-.75.75v8.5c0 .414.336.75.75.75h10.5a.75.75 0 00.75-.75v-8.5a.75.75 0 00-.75-.75H12zm.5 1.5v7h-9v-7h9zM5.5 4v2h5V4a2.5 2.5 0 00-5 0z",
      fill: "currentColor"
    })]
  });
}
function LockIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgLockIconV2 : SvgLockIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgLockUnlockedIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M18 8.5h-1v-2c0-2.76-2.24-5-5-5-2.28 0-4.27 1.54-4.84 3.75-.14.54.18 1.08.72 1.22a1 1 0 001.22-.72A2.996 2.996 0 0112 3.5c1.65 0 3 1.35 3 3v2H6c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2v-10c0-1.1-.9-2-2-2zm-6 5c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm5 7c.55 0 1-.45 1-1v-8c0-.55-.45-1-1-1H7c-.55 0-1 .45-1 1v8c0 .55.45 1 1 1h10z",
      fill: "currentColor"
    })
  });
}
function SvgLockUnlockedIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M10 11.75v-1.5H6v1.5h4z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M13.25 6H5.5V4a2.5 2.5 0 015 0v.5H12V4a4 4 0 00-8 0v2H2.75a.75.75 0 00-.75.75v8.5c0 .414.336.75.75.75h10.5a.75.75 0 00.75-.75v-8.5a.75.75 0 00-.75-.75zM3.5 7.5h9v7h-9v-7z",
      fill: "currentColor"
    })]
  });
}
function LockUnlockedIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgLockUnlockedIconV2 : SvgLockUnlockedIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgMIconV1(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M7.268 7.273V16h1.79v-5.702h.072l2.259 5.66h1.219l2.258-5.638h.073V16h1.79V7.273h-2.276l-2.403 5.863h-.103L9.544 7.273H7.268z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M17 3H7a4 4 0 00-4 4v10a4 4 0 004 4h10a4 4 0 004-4V7a4 4 0 00-4-4zM7 1a6 6 0 00-6 6v10a6 6 0 006 6h10a6 6 0 006-6V7a6 6 0 00-6-6H7z",
      fill: "currentColor"
    })]
  });
}
function SvgMIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M6.42 5.415A.75.75 0 005 5.75V11h1.5V8.927l.83 1.658a.75.75 0 001.34 0l.83-1.658V11H11V5.75a.75.75 0 00-1.42-.335L8 8.573 6.42 5.415z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M1.75 1a.75.75 0 00-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 00.75-.75V1.75a.75.75 0 00-.75-.75H1.75zm.75 12.5v-11h11v11h-11z",
      fill: "currentColor"
    })]
  });
}
function MIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgMIconV2 : SvgMIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgMinusBoxIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-1 16H6c-.55 0-1-.45-1-1V6c0-.55.45-1 1-1h12c.55 0 1 .45 1 1v12c0 .55-.45 1-1 1zm-9-5.5c-.55 0-1-.45-1-1s.45-1 1-1h6c.55 0 1 .45 1 1s-.45 1-1 1H9z",
      fill: "currentColor"
    })
  });
}
function SvgMinusBoxIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M11.5 8.75h-7v-1.5h7v1.5z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M1.75 1a.75.75 0 00-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 00.75-.75V1.75a.75.75 0 00-.75-.75H1.75zm.75 12.5v-11h11v11h-11z",
      fill: "currentColor"
    })]
  });
}
function MinusBoxIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgMinusBoxIconV2 : SvgMinusBoxIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgMinusCircleFillIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M2 12C2 6.48 6.48 2 12 2s10 4.48 10 10-4.48 10-10 10S2 17.52 2 12zm6-1c-.55 0-1 .45-1 1s.45 1 1 1h8c.55 0 1-.45 1-1s-.45-1-1-1H8z",
      fill: "currentColor"
    })
  });
}
function SvgMinusCircleFillIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M8 16A8 8 0 108 0a8 8 0 000 16zm3.5-7.25h-7v-1.5h7v1.5z",
      fill: "currentColor"
    })
  });
}
function MinusCircleFillIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgMinusCircleFillIconV2 : SvgMinusCircleFillIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgMinusCircleIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-4-9c-.55 0-1 .45-1 1s.45 1 1 1h8c.55 0 1-.45 1-1s-.45-1-1-1H8z",
      fill: "currentColor"
    })
  });
}
function SvgMinusCircleIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M4.5 8.75v-1.5h7v1.5h-7z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M0 8a8 8 0 1116 0A8 8 0 010 8zm8-6.5a6.5 6.5 0 100 13 6.5 6.5 0 000-13z",
      fill: "currentColor"
    })]
  });
}
function MinusCircleIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgMinusCircleIconV2 : SvgMinusCircleIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgModelsIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M20.876 6.87a3.001 3.001 0 10-3.815-3.475L6.741 5.777a3 3 0 10-2.519 4.215l2.636 7.907A3 3 0 1012 20l5.2-2.599a3 3 0 102.913-5.187l.763-5.343zM21 4a1 1 0 11-2 0 1 1 0 012 0zm-4.913 11.72l-4.764 2.382a2.994 2.994 0 00-2.65-1.084L6.094 9.283l9.993 4.997a3.006 3.006 0 000 1.44zM17.2 12.6L7.368 7.684l9.97-2.3c.327.628.87 1.126 1.53 1.395l-.765 5.357a2.99 2.99 0 00-.903.464zM4 8a1 1 0 100-2 1 1 0 000 2zm5 13a1 1 0 100-2 1 1 0 000 2zm10-5a1 1 0 100-2 1 1 0 000 2z",
      fill: "currentColor"
    })
  });
}
function SvgModelsIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("g", {
      clipPath: "url(#ModelsIcon_svg__clip0_13123_34951)",
      children: jsx("path", {
        fillRule: "evenodd",
        clipRule: "evenodd",
        d: "M0 4.75a2.75 2.75 0 015.145-1.353l4.372-.95a2.75 2.75 0 113.835 2.823l.282 2.257a2.75 2.75 0 11-2.517 4.46l-2.62 1.145.003.118a2.75 2.75 0 11-4.415-2.19L3.013 7.489A2.75 2.75 0 010 4.75zM2.75 3.5a1.25 1.25 0 100 2.5 1.25 1.25 0 000-2.5zm2.715 1.688c.018-.11.029-.22.033-.333l4.266-.928a2.753 2.753 0 002.102 1.546l.282 2.257c-.377.165-.71.412-.976.719L5.465 5.188zM4.828 6.55a2.767 2.767 0 01-.413.388l1.072 3.573a2.747 2.747 0 012.537 1.19l2.5-1.093a2.792 2.792 0 01.01-.797l-5.706-3.26zM12 10.25a1.25 1.25 0 112.5 0 1.25 1.25 0 01-2.5 0zM5.75 12a1.25 1.25 0 100 2.5 1.25 1.25 0 000-2.5zM11 2.75a1.25 1.25 0 112.5 0 1.25 1.25 0 01-2.5 0z",
        fill: "currentColor"
      })
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "ModelsIcon_svg__clip0_13123_34951",
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
function ModelsIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgModelsIconV2 : SvgModelsIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgNewWindowIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M5 18c0 .55.45 1 1 1h12c.55 0 1-.45 1-1v-5c0-.55.45-1 1-1s1 .45 1 1v6c0 1.1-.9 2-2 2H5c-1.1 0-2-.9-2-2V5a2 2 0 012-2h6c.55 0 1 .45 1 1s-.45 1-1 1H6c-.55 0-1 .45-1 1v12zM15 5c-.55 0-1-.45-1-1s.45-1 1-1h5c.55 0 1 .45 1 1v5c0 .55-.45 1-1 1s-1-.45-1-1V6.41l-9.13 9.13a.996.996 0 11-1.41-1.41L17.59 5H15z",
      fill: "currentColor"
    })
  });
}
function SvgNewWindowIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M10 1h5v5h-1.5V3.56L8.53 8.53 7.47 7.47l4.97-4.97H10V1z",
      fill: "currentColor"
    }), jsx("path", {
      d: "M1 2.75A.75.75 0 011.75 2H8v1.5H2.5v10h10V8H14v6.25a.75.75 0 01-.75.75H1.75a.75.75 0 01-.75-.75V2.75z",
      fill: "currentColor"
    })]
  });
}
function NewWindowIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgNewWindowIconV2 : SvgNewWindowIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgNoIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zM4 12c0-4.42 3.58-8 8-8 1.85 0 3.55.63 4.9 1.69L5.69 16.9A7.902 7.902 0 014 12zm3.1 6.31A7.902 7.902 0 0012 20c4.42 0 8-3.58 8-8 0-1.85-.63-3.55-1.69-4.9L7.1 18.31z",
      fill: "currentColor"
    })
  });
}
function SvgNoIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M8 0a8 8 0 100 16A8 8 0 008 0zM1.5 8a6.5 6.5 0 0110.535-5.096l-9.131 9.131A6.472 6.472 0 011.5 8zm2.465 5.096a6.5 6.5 0 009.131-9.131l-9.131 9.131z",
      fill: "currentColor"
    })
  });
}
function NoIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgNoIconV2 : SvgNoIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgNotebookIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M4 2v4H2v2h2v3H2v2h2v3H2v2h2v4h16a2 2 0 002-2V4a2 2 0 00-2-2H4zm4 2H6v2h2V4zm2 0v16h10V4H10zM8 20v-2H6v2h2zm0-4v-3H6v3h2zm0-5V8H6v3h2z",
      fill: "currentColor"
    })
  });
}
function SvgNotebookIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M3 1.75A.75.75 0 013.75 1h10.5a.75.75 0 01.75.75v12.5a.75.75 0 01-.75.75H3.75a.75.75 0 01-.75-.75V12.5H1V11h2V8.75H1v-1.5h2V5H1V3.5h2V1.75zm1.5.75v11H6v-11H4.5zm3 0v11h6v-11h-6z",
      fill: "currentColor"
    })
  });
}
function NotebookIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgNotebookIconV2 : SvgNotebookIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgNotificationIconV1(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M12 1.917c-3.5 0-6.5 3-6.5 6.083v5.278C5.5 16 3.5 18 3.5 18h17s-2-2-2-4.722V8c0-3.083-3-6.083-6.5-6.083z",
      stroke: "currentColor",
      strokeWidth: 2,
      strokeLinecap: "round",
      strokeLinejoin: "round"
    }), jsx("path", {
      d: "M9.5 21c.5.5 1.5 1 2.5 1s2-.5 2.5-1",
      stroke: "currentColor",
      strokeWidth: 2,
      strokeLinecap: "round"
    })]
  });
}
function SvgNotificationIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M8 1a5 5 0 00-5 5v1.99c0 .674-.2 1.332-.573 1.892l-1.301 1.952A.75.75 0 001.75 13h3.5v.25a2.75 2.75 0 105.5 0V13h3.5a.75.75 0 00.624-1.166l-1.301-1.952A3.41 3.41 0 0113 7.99V6a5 5 0 00-5-5zm1.25 12h-2.5v.25a1.25 1.25 0 102.5 0V13zM4.5 6a3.5 3.5 0 117 0v1.99c0 .97.287 1.918.825 2.724l.524.786H3.15l.524-.786A4.91 4.91 0 004.5 7.99V6z",
      fill: "currentColor"
    })
  });
}
function NotificationIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgNotificationIconV2 : SvgNotificationIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgOfficeIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M12 5v2h8c1.1 0 2 .9 2 2v10c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V5c0-1.1.9-2 2-2h6c1.1 0 2 .9 2 2zM4 19h2v-2H4v2zm2-4H4v-2h2v2zm-2-4h2V9H4v2zm2-4H4V5h2v2zm2 12h2v-2H8v2zm2-4H8v-2h2v2zm-2-4h2V9H8v2zm2-4H8V5h2v2zm2 12h7c.55 0 1-.45 1-1v-8c0-.55-.45-1-1-1h-7v2h2v2h-2v2h2v2h-2v2zm6-8h-2v2h2v-2zm-2 4h2v2h-2v-2z",
      fill: "currentColor"
    })
  });
}
function SvgOfficeIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M4 8.75h8v-1.5H4v1.5zM7 5.75H4v-1.5h3v1.5zM4 11.75h8v-1.5H4v1.5z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M1.75 1a.75.75 0 00-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 00.75-.75V5a.75.75 0 00-.75-.75H10v-2.5A.75.75 0 009.25 1h-7.5zm.75 1.5h6V5c0 .414.336.75.75.75h4.25v7.75h-11v-11z",
      fill: "currentColor"
    })]
  });
}
function OfficeIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgOfficeIconV2 : SvgOfficeIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgOverflowIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M12 8c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zm0 2c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm-2 8c0-1.1.9-2 2-2s2 .9 2 2-.9 2-2 2-2-.9-2-2z",
      fill: "currentColor"
    })
  });
}
function SvgOverflowIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M8 1a1.75 1.75 0 100 3.5A1.75 1.75 0 008 1zM8 6.25a1.75 1.75 0 100 3.5 1.75 1.75 0 000-3.5zM8 11.5A1.75 1.75 0 108 15a1.75 1.75 0 000-3.5z",
      fill: "currentColor"
    })
  });
}
function OverflowIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgOverflowIconV2 : SvgOverflowIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgPageBottomIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M17 18.001c.55 0 1-.45 1-1s-.45-1-1-1H7c-.55 0-1 .45-1 1s.45 1 1 1h10zm-5-7.82l3.89-3.89c.38-.38 1.02-.38 1.41 0a.996.996 0 010 1.41l-4.6 4.59a.996.996 0 01-1.41 0L6.7 7.701a.996.996 0 111.41-1.41l3.89 3.89z",
      fill: "currentColor"
    })
  });
}
function SvgPageBottomIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M1 3.06L2.06 2l5.97 5.97L14 2l1.06 1.06-7.03 7.031L1 3.061zm14.03 10.47v1.5h-14v-1.5h14z",
      fill: "currentColor"
    })
  });
}
function PageBottomIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgPageBottomIconV2 : SvgPageBottomIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgPageFirstIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M7.999 7c0-.55-.45-1-1-1s-1 .45-1 1v10c0 .55.45 1 1 1s1-.45 1-1V7zm5.82 5l3.88 3.89c.39.38.39 1.02.01 1.4a.996.996 0 01-1.41 0l-4.59-4.59a.996.996 0 010-1.41l4.59-4.59a.996.996 0 111.41 1.41L13.819 12z",
      fill: "currentColor"
    })
  });
}
function SvgPageFirstIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M12.97 1l1.06 1.06-5.97 5.97L14.03 14l-1.06 1.06-7.03-7.03L12.97 1zM2.5 15.03H1v-14h1.5v14z",
      fill: "currentColor"
    })
  });
}
function PageFirstIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgPageFirstIconV2 : SvgPageFirstIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgPageLastIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M18.001 7c0-.55-.45-1-1-1s-1 .45-1 1v10c0 .55.45 1 1 1s1-.45 1-1V7zm-7.82 5l-3.89-3.89c-.38-.38-.38-1.02 0-1.41a.996.996 0 011.41 0l4.59 4.6c.39.39.39 1.02 0 1.41l-4.59 4.59a.996.996 0 11-1.41-1.41l3.89-3.89z",
      fill: "currentColor"
    })
  });
}
function SvgPageLastIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M3.06 1L2 2.06l5.97 5.97L2 14l1.06 1.06 7.031-7.03L3.061 1zm10.47 14.03h1.5v-14h-1.5v14z",
      fill: "currentColor"
    })
  });
}
function PageLastIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgPageLastIconV2 : SvgPageLastIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgPageTopIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M7 5.999c-.55 0-1 .45-1 1s.45 1 1 1h10c.55 0 1-.45 1-1s-.45-1-1-1H7zm5 7.82l-3.89 3.89c-.38.38-1.02.38-1.41 0a.996.996 0 010-1.41l4.6-4.59a.996.996 0 011.41 0l4.59 4.59a.996.996 0 11-1.41 1.41L12 13.819z",
      fill: "currentColor"
    })
  });
}
function SvgPageTopIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M1 12.97l1.06 1.06 5.97-5.97L14 14.03l1.06-1.06-7.03-7.03L1 12.97zM15.03 2.5V1h-14v1.5h14z",
      fill: "currentColor"
    })
  });
}
function PageTopIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgPageTopIconV2 : SvgPageTopIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgPencilIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M20.709 5.631c.39.39.39 1.02 0 1.41l-1.83 1.83-3.75-3.75 1.83-1.83a.996.996 0 011.41 0l2.34 2.34zm-17.71 14.87v-3.04c0-.14.05-.26.15-.36l10.91-10.91 3.75 3.75-10.92 10.91a.47.47 0 01-.35.15h-3.04c-.28 0-.5-.22-.5-.5z",
      fill: "currentColor"
    })
  });
}
function SvgPencilIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M13.487 1.513a1.75 1.75 0 00-2.474 0L1.22 11.306a.75.75 0 00-.22.53v2.5c0 .414.336.75.75.75h2.5a.75.75 0 00.53-.22l9.793-9.793a1.75 1.75 0 000-2.475l-1.086-1.085zm-1.414 1.06a.25.25 0 01.354 0l1.086 1.086a.25.25 0 010 .354L12 5.525l-1.44-1.44 1.513-1.512zM9.5 5.146l-7 7v1.44h1.44l7-7-1.44-1.44z",
      fill: "currentColor"
    })
  });
}
function PencilIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgPencilIconV2 : SvgPencilIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgPinCancelIconV1(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M12 22v-5",
      stroke: "currentColor",
      strokeWidth: 2,
      strokeLinecap: "round"
    }), jsx("path", {
      d: "M6.5 17h11c1.236 0 1.942-1.411 1.2-2.4L16 11V5a2 2 0 00-2-2h-4a2 2 0 00-2 2v6l-2.7 3.6c-.742.989-.036 2.4 1.2 2.4z",
      fill: "currentColor",
      stroke: "currentColor",
      strokeWidth: 2
    }), jsx("path", {
      d: "M10 14l4-4M10 10l4 4",
      stroke: "#fff",
      strokeWidth: 2,
      strokeLinecap: "round"
    })]
  });
}
function SvgPinCancelIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M5.75 0A.75.75 0 005 .75v1.19l9 9V9a.75.75 0 00-.22-.53l-2.12-2.122a2.25 2.25 0 01-.66-1.59V.75a.75.75 0 00-.75-.75h-4.5zM10.94 12l2.53 2.53 1.06-1.06-11.5-11.5-1.06 1.06 2.772 2.773c-.104.2-.239.383-.4.545L2.22 8.47A.75.75 0 002 9v2.25c0 .414.336.75.75.75h4.5v4h1.5v-4h2.19z",
      fill: "currentColor"
    })
  });
}
function PinCancelIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgPinCancelIconV2 : SvgPinCancelIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgPinFillIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M7 5a3 3 0 013-3h4a3 3 0 013 3v5.667L19.5 14c1.236 1.648.06 4-2 4H13v4a1 1 0 01-2 0v-4H6.5c-2.06 0-3.236-2.352-2-4L7 10.667V5z",
      fill: "currentColor"
    })
  });
}
function SvgPinFillIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M5 .75A.75.75 0 015.75 0h4.5a.75.75 0 01.75.75v4.007c0 .597.237 1.17.659 1.591L13.78 8.47c.141.14.22.331.22.53v2.25a.75.75 0 01-.75.75h-4.5v4h-1.5v-4h-4.5a.75.75 0 01-.75-.75V9a.75.75 0 01.22-.53L4.34 6.348A2.25 2.25 0 005 4.758V.75z",
      fill: "currentColor"
    })
  });
}
function PinFillIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgPinFillIconV2 : SvgPinFillIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgPinIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M10 2a3 3 0 00-3 3v5.667L4.5 14c-1.236 1.648-.06 4 2 4H11v4a1 1 0 102 0v-4h4.5c2.06 0 3.236-2.352 2-4L17 10.667V5a3 3 0 00-3-3h-4zm2.004 14H17.5a.5.5 0 00.4-.8L15 11.333V5a1 1 0 00-1-1h-4a1 1 0 00-1 1v6.333L6.1 15.2a.5.5 0 00.4.8h5.504z",
      fill: "currentColor"
    })
  });
}
function SvgPinIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M5.75 0A.75.75 0 005 .75v4.007a2.25 2.25 0 01-.659 1.591L2.22 8.47A.75.75 0 002 9v2.25c0 .414.336.75.75.75h4.5v4h1.5v-4h4.5a.75.75 0 00.75-.75V9a.75.75 0 00-.22-.53L11.66 6.348A2.25 2.25 0 0111 4.758V.75a.75.75 0 00-.75-.75h-4.5zm.75 4.757V1.5h3v3.257a3.75 3.75 0 001.098 2.652L12.5 9.311V10.5h-9V9.31L5.402 7.41A3.75 3.75 0 006.5 4.757z",
      fill: "currentColor"
    })
  });
}
function PinIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgPinIconV2 : SvgPinIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgPlayCircleFillIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M2 12C2 6.48 6.48 2 12 2s10 4.48 10 10-4.48 10-10 10S2 17.52 2 12zm8.8-3.9a.5.5 0 00-.8.4v7c0 .41.47.65.8.4l4.67-3.5c.27-.2.27-.6 0-.8L10.8 8.1z",
      fill: "currentColor"
    })
  });
}
function SvgPlayCircleFillIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M0 8a8 8 0 1116 0A8 8 0 010 8zm7.125-2.815A.75.75 0 006 5.835v4.33a.75.75 0 001.125.65l3.75-2.166a.75.75 0 000-1.299l-3.75-2.165z",
      fill: "currentColor"
    })
  });
}
function PlayCircleFillIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgPlayCircleFillIconV2 : SvgPlayCircleFillIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgPlayCircleIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1.2 13.9l4.67-3.5c.27-.2.27-.6 0-.8L10.8 8.1a.5.5 0 00-.8.4v7c0 .41.47.65.8.4zM4 12c0 4.41 3.59 8 8 8s8-3.59 8-8-3.59-8-8-8-8 3.59-8 8z",
      fill: "currentColor"
    })
  });
}
function SvgPlayCircleIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M11.25 8a.75.75 0 01-.375.65l-3.75 2.165A.75.75 0 016 10.165v-4.33a.75.75 0 011.125-.65l3.75 2.165a.75.75 0 01.375.65z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M8 0a8 8 0 100 16A8 8 0 008 0zM1.5 8a6.5 6.5 0 1113 0 6.5 6.5 0 01-13 0z",
      fill: "currentColor"
    })]
  });
}
function PlayCircleIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgPlayCircleIconV2 : SvgPlayCircleIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgPlayIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M6.927 6.82v10.36c0 .79.87 1.27 1.54.84l8.14-5.18a1 1 0 000-1.69l-8.14-5.17a.998.998 0 00-1.54.84z",
      fill: "currentColor"
    })
  });
}
function SvgPlayIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M12.125 8.864a.75.75 0 000-1.3l-6-3.464A.75.75 0 005 4.75v6.928a.75.75 0 001.125.65l6-3.464z",
      fill: "currentColor"
    })
  });
}
function PlayIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgPlayIconV2 : SvgPlayIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgPlusCircleFillIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M2 12C2 6.48 6.48 2 12 2s10 4.48 10 10-4.48 10-10 10S2 17.52 2 12zm11 1h3c.55 0 1-.45 1-1s-.45-1-1-1h-3V8c0-.55-.45-1-1-1s-1 .45-1 1v3H8c-.55 0-1 .45-1 1s.45 1 1 1h3v3c0 .55.45 1 1 1s1-.45 1-1v-3z",
      fill: "currentColor"
    })
  });
}
function SvgPlusCircleFillIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M8 16A8 8 0 108 0a8 8 0 000 16zm-.75-4.5V8.75H4.5v-1.5h2.75V4.5h1.5v2.75h2.75v1.5H8.75v2.75h-1.5z",
      fill: "currentColor"
    })
  });
}
function PlusCircleFillIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgPlusCircleFillIconV2 : SvgPlusCircleFillIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgPlusCircleIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 5c-.55 0-1 .45-1 1v3H8c-.55 0-1 .45-1 1s.45 1 1 1h3v3c0 .55.45 1 1 1s1-.45 1-1v-3h3c.55 0 1-.45 1-1s-.45-1-1-1h-3V8c0-.55-.45-1-1-1zm-8 5c0 4.41 3.59 8 8 8s8-3.59 8-8-3.59-8-8-8-8 3.59-8 8z",
      fill: "currentColor"
    })
  });
}
function SvgPlusCircleIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M7.25 11.5V8.75H4.5v-1.5h2.75V4.5h1.5v2.75h2.75v1.5H8.75v2.75h-1.5z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M8 0a8 8 0 100 16A8 8 0 008 0zM1.5 8a6.5 6.5 0 1113 0 6.5 6.5 0 01-13 0z",
      fill: "currentColor"
    })]
  });
}
function PlusCircleIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgPlusCircleIconV2 : SvgPlusCircleIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgPlusIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M18 13h-5v5c0 .55-.45 1-1 1s-1-.45-1-1v-5H6c-.55 0-1-.45-1-1s.45-1 1-1h5V6c0-.55.45-1 1-1s1 .45 1 1v5h5c.55 0 1 .45 1 1s-.45 1-1 1z",
      fill: "currentColor"
    })
  });
}
function SvgPlusIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M7.25 7.25V1h1.5v6.25H15v1.5H8.75V15h-1.5V8.75H1v-1.5h6.25z",
      fill: "currentColor"
    })
  });
}
function PlusIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgPlusIconV2 : SvgPlusIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgPlusSquareIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 15c0 .55-.45 1-1 1H6c-.55 0-1-.45-1-1V6c0-.55.45-1 1-1h12c.55 0 1 .45 1 1v12zM11 8c0-.55.45-1 1-1s1 .45 1 1v3h3c.55 0 1 .45 1 1s-.45 1-1 1h-3v3c0 .55-.45 1-1 1s-1-.45-1-1v-3H8c-.55 0-1-.45-1-1s.45-1 1-1h3V8z",
      fill: "currentColor"
    })
  });
}
function SvgPlusSquareIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M7.25 7.25V4.5h1.5v2.75h2.75v1.5H8.75v2.75h-1.5V8.75H4.5v-1.5h2.75z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M1 1.75A.75.75 0 011.75 1h12.5a.75.75 0 01.75.75v12.5a.75.75 0 01-.75.75H1.75a.75.75 0 01-.75-.75V1.75zm1.5.75v11h11v-11h-11z",
      fill: "currentColor"
    })]
  });
}
function PlusSquareIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgPlusSquareIconV2 : SvgPlusSquareIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgQueryEditorIconV1(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M2 19c0 1.652 1.348 3 3 3h14c1.652 0 3-1.348 3-3V5c0-1.652-1.348-3-3-3H5C3.348 2 2 3.348 2 5v14zM5 4c-.548 0-1 .452-1 1v3h16V5c0-.548-.452-1-1-1H5zm15 6H4v9c0 .548.452 1 1 1h14c.548 0 1-.452 1-1v-9z",
      fill: "currentColor"
    }), jsx("path", {
      d: "M7.293 12.293a1 1 0 011.414 0l2 2a1 1 0 010 1.414l-2 2a1 1 0 01-1.414-1.414L8.586 15l-1.293-1.293a1 1 0 010-1.414z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M12 17a1 1 0 011-1h4a1 1 0 110 2h-4a1 1 0 01-1-1z",
      fill: "currentColor"
    })]
  });
}
function SvgQueryEditorIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M12 12H8v-1.5h4V12zM5.53 11.53L7.56 9.5 5.53 7.47 4.47 8.53l.97.97-.97.97 1.06 1.06z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M1.75 1a.75.75 0 00-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 00.75-.75V1.75a.75.75 0 00-.75-.75H1.75zm.75 3V2.5h11V4h-11zm0 1.5v8h11v-8h-11z",
      fill: "currentColor"
    })]
  });
}
function QueryEditorIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgQueryEditorIconV2 : SvgQueryEditorIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgQueryIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M19.88 18.47c.44-.7.7-1.51.7-2.39 0-2.49-2.01-4.5-4.5-4.5s-4.5 2.01-4.5 4.5 2.01 4.5 4.49 4.5c.88 0 1.7-.26 2.39-.7L21.58 23 23 21.58l-3.12-3.11zm-3.8.11a2.5 2.5 0 010-5 2.5 2.5 0 010 5zm-.36-8.5c-.74.02-1.45.18-2.1.45l-.55-.83-3.8 6.18-3.01-3.52-3.63 5.81L1 17l5-8 3 3.5L13 6l2.72 4.08zm2.59.5c-.64-.28-1.33-.45-2.05-.49L21.38 2 23 3.18l-4.69 7.4z",
      fill: "currentColor"
    })
  });
}
function SvgQueryIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsxs("g", {
      clipPath: "url(#QueryIcon_svg__clip0_13123_35183)",
      fill: "currentColor",
      children: [jsx("path", {
        fillRule: "evenodd",
        clipRule: "evenodd",
        d: "M2 1.75A.75.75 0 012.75 1h6a.75.75 0 01.53.22l4.5 4.5c.141.14.22.331.22.53V10h-1.5V7H8.75A.75.75 0 018 6.25V2.5H3.5V16h-.75a.75.75 0 01-.75-.75V1.75zm7.5 1.81l1.94 1.94H9.5V3.56z"
      }), jsx("path", {
        d: "M5.53 9.97L8.56 13l-3.03 3.03-1.06-1.06L6.44 13l-1.97-1.97 1.06-1.06zM14 14.5H9V16h5v-1.5z"
      })]
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "QueryIcon_svg__clip0_13123_35183",
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
function QueryIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgQueryIconV2 : SvgQueryIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgQuestionMarkFillIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17v-2h2v2h-2zm3.17-6.83l.9-.92c1.02-1.02 1.37-2.77.19-4.4-.9-1.25-2.35-2.04-3.87-1.8-1.55.24-2.8 1.36-3.23 2.83C8 8.44 8.4 9 8.98 9h.3c.39 0 .7-.28.82-.65.33-.95 1.36-1.58 2.47-1.27.7.2 1.26.81 1.39 1.53.13.7-.09 1.36-.55 1.8l-1.24 1.26A3.997 3.997 0 0011 14.5v.5h2c0-.46.05-.82.13-1.14.18-.72.54-1.18 1.04-1.69z",
      fill: "currentColor"
    })
  });
}
function SvgQuestionMarkFillIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M8 16A8 8 0 108 0a8 8 0 000 16zm2.207-10.189a2.25 2.25 0 01-1.457 2.56V9h-1.5V7.75A.75.75 0 018 7a.75.75 0 10-.75-.75h-1.5a2.25 2.25 0 014.457-.439zM7.25 10.75a.75.75 0 101.5 0 .75.75 0 00-1.5 0z",
      fill: "currentColor"
    })
  });
}
function QuestionMarkFillIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgQuestionMarkFillIconV2 : SvgQuestionMarkFillIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgQuestionMarkIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M2 12C2 6.48 6.48 2 12 2s10 4.48 10 10-4.48 10-10 10S2 17.52 2 12zm2 0c0 4.41 3.59 8 8 8s8-3.59 8-8-3.59-8-8-8-8 3.59-8 8zm9 4v2h-2v-2h2zM8.18 8.83a4.002 4.002 0 014.43-2.79c1.74.26 3.11 1.73 3.35 3.47.228 1.614-.664 2.392-1.526 3.143-.158.139-.315.276-.464.417-.12.11-.23.22-.33.34-.005.005-.01.012-.015.02l-.015.02a2.758 2.758 0 00-.33.48c-.17.3-.28.65-.28 1.07h-2c0-.5.08-.91.2-1.25l.01-.034a.144.144 0 01.01-.036c.005-.015.012-.027.02-.04.008-.013.015-.025.02-.04a3.331 3.331 0 01.265-.525l.015-.025c0-.005.003-.008.005-.01s.005-.005.005-.01c.34-.513.797-.864 1.224-1.193.614-.472 1.167-.897 1.226-1.687.08-.97-.62-1.9-1.57-2.1-1.03-.22-1.98.39-2.3 1.28-.14.38-.47.67-.88.67h-.2a.907.907 0 01-.87-1.17z",
      fill: "currentColor"
    })
  });
}
function SvgQuestionMarkIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M7.25 10.75a.75.75 0 101.5 0 .75.75 0 00-1.5 0zM10.079 7.111A2.25 2.25 0 105.75 6.25h1.5A.75.75 0 118 7a.75.75 0 00-.75.75V9h1.5v-.629a2.25 2.25 0 001.329-1.26z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M0 8a8 8 0 1116 0A8 8 0 010 8zm8-6.5a6.5 6.5 0 100 13 6.5 6.5 0 000-13z",
      fill: "currentColor"
    })]
  });
}
function QuestionMarkIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgQuestionMarkIconV2 : SvgQuestionMarkIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgReaderModeIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M21 3.5H3c-1.1 0-2 .9-2 2v13c0 1.1.9 2 2 2h18c1.1 0 2-.9 2-2v-13c0-1.1-.9-2-2-2zm0 14c0 .55-.45 1-1 1h-8v-13h8c.55 0 1 .45 1 1v11zm-6.5-10a1 1 0 100 2h4a1 1 0 100-2h-4zm-1 4.5a1 1 0 011-1h4a1 1 0 110 2h-4a1 1 0 01-1-1zm1 2.5a1 1 0 100 2h4a1 1 0 100-2h-4z",
      fill: "currentColor"
    })
  });
}
function SvgReaderModeIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M13 4.5h-3V6h3V4.5zM13 7.25h-3v1.5h3v-1.5zM13 10h-3v1.5h3V10z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M.75 2a.75.75 0 00-.75.75v10.5c0 .414.336.75.75.75h14.5a.75.75 0 00.75-.75V2.75a.75.75 0 00-.75-.75H.75zm.75 10.5v-9h5.75v9H1.5zm7.25 0h5.75v-9H8.75v9z",
      fill: "currentColor"
    })]
  });
}
function ReaderModeIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgReaderModeIconV2 : SvgReaderModeIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgRedoIconV1(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("g", {
      clipPath: "url(#RedoIcon_svg__clip0_13917_34618)",
      children: jsx("path", {
        d: "M13.129 6.5h-8.69a3 3 0 100 6h4.5V14h-4.5a4.5 4.5 0 010-9h8.69l-2.72-2.72 1.06-1.06L16 5.75l-4.53 4.53-1.061-1.06 2.72-2.72z",
        fill: "currentColor"
      })
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "RedoIcon_svg__clip0_13917_34618",
        children: jsx("path", {
          fill: "#fff",
          transform: "rotate(-180 8 8)",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
function SvgRedoIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("g", {
      clipPath: "url(#RedoIcon_svg__clip0_14136_38626)",
      children: jsx("path", {
        fillRule: "evenodd",
        clipRule: "evenodd",
        d: "M13.19 5l-2.72-2.72 1.06-1.06 4.53 4.53-4.53 4.53-1.06-1.06 2.72-2.72H4.5a3 3 0 100 6H9V14H4.5a4.5 4.5 0 010-9h8.69z",
        fill: "currentColor"
      })
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "RedoIcon_svg__clip0_14136_38626",
        children: jsx("path", {
          fill: "#fff",
          transform: "matrix(1 0 0 -1 0 16)",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
function RedoIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgRedoIconV2 : SvgRedoIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgRefreshIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M8 2.5a5.48 5.48 0 013.817 1.54l.009.009.5.451H11V6h4V2h-1.5v1.539l-.651-.588a7 7 0 10.114 9.985l-1.064-1.057A5.5 5.5 0 118 2.5z",
      fill: "currentColor"
    })
  });
}
function SvgRefreshIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M1 8a7 7 0 0111.85-5.047l.65.594V2H15v4h-4V4.5h1.32l-.496-.453-.007-.007a5.5 5.5 0 10.083 7.839l1.063 1.058A7 7 0 011 8z",
      fill: "currentColor"
    })
  });
}
function RefreshIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgRefreshIconV2 : SvgRefreshIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgReposIconV1(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M4 5a1 1 0 011-1h3.93a1 1 0 01.832.445L11.465 7H19a1 1 0 011 1v3h2V8a3 3 0 00-3-3h-6.465l-1.11-1.664A3 3 0 008.93 2H5a3 3 0 00-3 3v14a3 3 0 003 3h4v-2H5a1 1 0 01-1-1V5z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M17 12c0 .69-.234 1.327-.626 1.834.376.774.803 1.137 1.198 1.334.265.132.55.212.872.26a3 3 0 11-.42 1.957 4.831 4.831 0 01-1.346-.428c-.653-.327-1.214-.817-1.678-1.507v1.72a3.001 3.001 0 11-2 0v-2.34A3.001 3.001 0 1117 12zm-2 0a1 1 0 11-2 0 1 1 0 012 0zm6 6a1 1 0 110-2 1 1 0 010 2zm-6 2a1 1 0 11-2 0 1 1 0 012 0z",
      fill: "currentColor"
    })]
  });
}
function SvgReposIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M0 2.75A.75.75 0 01.75 2h3.922c.729 0 1.428.29 1.944.805L7.811 4h7.439a.75.75 0 01.75.75V8h-1.5V5.5h-7a.75.75 0 01-.53-.22L5.555 3.866a1.25 1.25 0 00-.883-.366H1.5v9H5V14H.75a.75.75 0 01-.75-.75V2.75zM9 8.5a.5.5 0 100 1 .5.5 0 000-1zM7 9a2 2 0 113.778.917c.376.58.888 1.031 1.414 1.227a2 2 0 11-.072 1.54c-.977-.207-1.795-.872-2.37-1.626v1.087a2 2 0 11-1.5 0v-1.29A2 2 0 017 9zm7 2.5a.5.5 0 100 1 .5.5 0 000-1zm-5 2a.5.5 0 100 1 .5.5 0 000-1z",
      fill: "currentColor"
    })
  });
}
function ReposIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgReposIconV2 : SvgReposIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgSchoolIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M2.064 8.116l8.43-4.6c.6-.32 1.32-.32 1.92 0l9.52 5.19c.32.18.52.51.52.88v6.41c0 .55-.45 1-1 1s-1-.45-1-1v-5.91l-8.04 4.39c-.6.33-1.32.33-1.92 0l-8.43-4.6c-.69-.38-.69-1.38 0-1.76zm2.39 7.87v-2.81l6.04 3.3c.6.33 1.32.33 1.92 0l6.04-3.3v2.81c0 .73-.4 1.41-1.04 1.76l-5 2.73c-.6.33-1.32.33-1.92 0l-5-2.73a2.011 2.011 0 01-1.04-1.76z",
      fill: "currentColor"
    })
  });
}
function SvgSchoolIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M16 7a.75.75 0 00-.37-.647l-7.25-4.25a.75.75 0 00-.76 0L.37 6.353a.75.75 0 000 1.294L3 9.188V12a.75.75 0 00.4.663l4.25 2.25a.75.75 0 00.7 0l4.25-2.25A.75.75 0 0013 12V9.188l1.5-.879V12H16V7zm-7.62 4.897l3.12-1.83v1.481L8 13.401l-3.5-1.853v-1.48l3.12 1.829a.75.75 0 00.76 0zM8 3.619L2.233 7 8 10.38 13.767 7 8 3.62z",
      fill: "currentColor"
    })
  });
}
function SchoolIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgSchoolIconV2 : SvgSchoolIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgSearchIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M15.187 14.472h.79l4.24 4.26c.41.41.41 1.08 0 1.49-.41.41-1.08.41-1.49 0l-4.25-4.25v-.79l-.27-.28a6.5 6.5 0 01-5.34 1.48c-2.78-.47-5-2.79-5.34-5.59a6.505 6.505 0 017.27-7.27c2.8.34 5.12 2.56 5.59 5.34a6.5 6.5 0 01-1.48 5.34l.28.27zm-9.71-4.5c0 2.49 2.01 4.5 4.5 4.5s4.5-2.01 4.5-4.5-2.01-4.5-4.5-4.5-4.5 2.01-4.5 4.5z",
      fill: "currentColor"
    })
  });
}
function SvgSearchIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("g", {
      clipPath: "url(#SearchIcon_svg__clip0_13123_34883)",
      children: jsx("path", {
        fillRule: "evenodd",
        clipRule: "evenodd",
        d: "M8 1a7 7 0 104.39 12.453l2.55 2.55 1.06-1.06-2.55-2.55A7 7 0 008 1zM2.5 8a5.5 5.5 0 1111 0 5.5 5.5 0 01-11 0z",
        fill: "currentColor"
      })
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "SearchIcon_svg__clip0_13123_34883",
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
function SearchIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgSearchIconV2 : SvgSearchIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgSecurityIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M4.19 4.376l7-3.11c.51-.23 1.11-.23 1.62 0l7 3.11c.72.32 1.19 1.04 1.19 1.83v4.7c0 5.55-3.84 10.74-9 12-5.16-1.26-9-6.45-9-12v-4.7c0-.79.47-1.51 1.19-1.83zM19 11.896h-7v-8.8l-7 3.11v5.7h7v8.93c3.72-1.15 6.47-4.82 7-8.94z",
      fill: "currentColor"
    })
  });
}
function SvgSecurityIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M2 1.75A.75.75 0 012.75 1h10.5a.75.75 0 01.75.75v7.465a5.75 5.75 0 01-2.723 4.889l-2.882 1.784a.75.75 0 01-.79 0l-2.882-1.784A5.75 5.75 0 012 9.214V1.75zm1.5.75V7h3.75V2.5H3.5zm5.25 0V7h3.75V2.5H8.75zm3.75 6H8.75v5.404l1.737-1.076A4.25 4.25 0 0012.5 9.215V8.5zm-5.25 5.404V8.5H3.5v.715a4.25 4.25 0 002.013 3.613l1.737 1.076z",
      fill: "currentColor"
    })
  });
}
function SecurityIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgSecurityIconV2 : SvgSecurityIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgShareIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M18 16.12c-.76 0-1.44.3-1.96.77l-7.13-4.15c.05-.23.09-.46.09-.7 0-.24-.04-.47-.09-.7l7.05-4.11c.54.5 1.25.81 2.04.81 1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3c0 .24.04.47.09.7L8.04 9.85c-.54-.5-1.25-.81-2.04-.81-1.66 0-3 1.34-3 3s1.34 3 3 3c.79 0 1.5-.31 2.04-.81l7.12 4.16c-.05.21-.08.43-.08.65 0 1.61 1.31 2.92 2.92 2.92 1.61 0 2.92-1.31 2.92-2.92 0-1.61-1.31-2.92-2.92-2.92z",
      fill: "currentColor"
    })
  });
}
function SvgShareIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M3.97 5.03L8 1l4.03 4.03-1.06 1.061-2.22-2.22v7.19h-1.5V3.87l-2.22 2.22-1.06-1.06z",
      fill: "currentColor"
    }), jsx("path", {
      d: "M2.5 13.56v-6.5H1v7.25c0 .415.336.75.75.75h12.5a.75.75 0 00.75-.75V7.06h-1.5v6.5h-11z",
      fill: "currentColor"
    })]
  });
}
function ShareIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgShareIconV2 : SvgShareIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgSidebarAutoIconV1(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M10 20h10a1 1 0 110 2H5c-1.652 0-3-1.348-3-3V5c0-1.652 1.348-3 3-3h15a1 1 0 110 2H10v16zm-2 0H5c-.548 0-1-.452-1-1V5c0-.548.452-1 1-1h3v16z",
      fill: "currentColor"
    }), jsx("path", {
      d: "M19.707 8.293a1 1 0 10-1.414 1.414L20.586 12l-2.293 2.293a1 1 0 001.414 1.414l3-3a1 1 0 000-1.414l-3-3zM15.707 15.707a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414l3-3a1 1 0 111.414 1.414L13.414 12l2.293 2.293a1 1 0 010 1.414z",
      fill: "currentColor"
    })]
  });
}
function SvgSidebarAutoIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 17 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M1.75 1a.75.75 0 00-.75.75v12.5c0 .414.336.75.75.75H15v-1.5H5.5v-11H15V1H1.75zM4 2.5H2.5v11H4v-11z",
      fill: "currentColor"
    }), jsx("path", {
      d: "M9.06 8l1.97 1.97-1.06 1.06L6.94 8l3.03-3.03 1.06 1.06L9.06 8zM11.97 6.03L13.94 8l-1.97 1.97 1.06 1.06L16.06 8l-3.03-3.03-1.06 1.06z",
      fill: "currentColor"
    })]
  });
}
function SidebarAutoIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgSidebarAutoIconV2 : SvgSidebarAutoIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgSidebarCollapseIconV1(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M20 20H10V4h10a1 1 0 100-2H5C3.348 2 2 3.348 2 5v14c0 1.652 1.348 3 3 3h15a1 1 0 100-2zM5 20h3V4H5c-.548 0-1 .452-1 1v14c0 .548.452 1 1 1z",
      fill: "currentColor"
    }), jsx("path", {
      d: "M17.707 16.707a1 1 0 01-1.414 0L12 12c0-.277.112-.527.294-.708l3.999-4a1 1 0 111.414 1.415L15.414 11H21a1 1 0 110 2h-5.586l2.293 2.293a1 1 0 010 1.414z",
      fill: "currentColor"
    })]
  });
}
function SvgSidebarCollapseIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M1.75 1a.75.75 0 00-.75.75v12.5c0 .414.336.75.75.75H15v-1.5H5.5v-11H15V1H1.75zM4 2.5H2.5v11H4v-11z",
      fill: "currentColor"
    }), jsx("path", {
      d: "M9.81 8.75l1.22 1.22-1.06 1.06L6.94 8l3.03-3.03 1.06 1.06-1.22 1.22H14v1.5H9.81z",
      fill: "currentColor"
    })]
  });
}
function SidebarCollapseIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgSidebarCollapseIconV2 : SvgSidebarCollapseIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgSidebarExpandIconV1(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M10 20h10a1 1 0 110 2H5c-1.652 0-3-1.348-3-3V5c0-1.652 1.348-3 3-3h15a1 1 0 110 2H10v16zm-2 0V4H5c-.548 0-1 .452-1 1v14c0 .548.452 1 1 1h3z",
      fill: "currentColor"
    }), jsx("path", {
      d: "M17.707 7.293a1 1 0 10-1.414 1.414L18.586 11H13a1 1 0 100 2h5.586l-2.293 2.293a1 1 0 001.414 1.414l4-3.999.007-.007a.997.997 0 00.286-.698v-.006a.996.996 0 00-.293-.704l-4-4z",
      fill: "currentColor"
    })]
  });
}
function SvgSidebarExpandIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M1.75 1a.75.75 0 00-.75.75v12.5c0 .414.336.75.75.75H15v-1.5H5.5v-11H15V1H1.75zM4 2.5H2.5v11H4v-11z",
      fill: "currentColor"
    }), jsx("path", {
      d: "M11.19 8.75L9.97 9.97l1.06 1.06L14.06 8l-3.03-3.03-1.06 1.06 1.22 1.22H7v1.5h4.19z",
      fill: "currentColor"
    })]
  });
}
function SidebarExpandIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgSidebarExpandIconV2 : SvgSidebarExpandIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgSidebarIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M5 22c-1.652 0-3-1.348-3-3V5c0-1.652 1.348-3 3-3h14c1.652 0 3 1.348 3 3v14c0 1.652-1.348 3-3 3H5zM4 5c0-.548.452-1 1-1h3v16H5c-.548 0-1-.452-1-1V5zm6 15V4h9c.548 0 1 .452 1 1v14c0 .548-.452 1-1 1h-9z",
      fill: "currentColor"
    })
  });
}
function SvgSidebarIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M1.75 1a.75.75 0 00-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 00.75-.75V1.75a.75.75 0 00-.75-.75H1.75zm.75 12.5v-11H4v11H2.5zm3 0h8v-11h-8v11z",
      fill: "currentColor"
    })
  });
}
function SidebarIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgSidebarIconV2 : SvgSidebarIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgSpeechBubbleIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M20 2c1.1 0 1.99.9 1.99 2L22 22l-4-4H4c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h16zM7 14h10c.55 0 1-.45 1-1s-.45-1-1-1H7c-.55 0-1 .45-1 1s.45 1 1 1zm10-3H7c-.55 0-1-.45-1-1s.45-1 1-1h10c.55 0 1 .45 1 1s-.45 1-1 1zM7 8h10c.55 0 1-.45 1-1s-.45-1-1-1H7c-.55 0-1 .45-1 1s.45 1 1 1z",
      fill: "currentColor"
    })
  });
}
function SvgSpeechBubbleIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M8 8.75a.75.75 0 100-1.5.75.75 0 000 1.5zM11.5 8A.75.75 0 1110 8a.75.75 0 011.5 0zM5.25 8.75a.75.75 0 100-1.5.75.75 0 000 1.5z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M8 15c-.099 0-.197-.002-.295-.006A.762.762 0 017.61 15H1.75a.75.75 0 01-.53-1.28l1.328-1.329A7 7 0 118 15zM2.5 8a5.5 5.5 0 115.156 5.49.75.75 0 00-.18.01H3.56l.55-.55a.75.75 0 000-1.06A5.48 5.48 0 012.5 8z",
      fill: "currentColor"
    })]
  });
}
function SpeechBubbleIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgSpeechBubbleIconV2 : SvgSpeechBubbleIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgSpeechBubblePlusIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M20 2c1.1 0 2 .9 2 2v18l-4-4H4c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h16zm-7 9h3c.55 0 1-.45 1-1s-.45-1-1-1h-3V6c0-.55-.45-1-1-1s-1 .45-1 1v3H8c-.55 0-1 .45-1 1s.45 1 1 1h3v3c0 .55.45 1 1 1s1-.45 1-1v-3z",
      fill: "currentColor"
    })
  });
}
function SvgSpeechBubblePlusIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M7.25 11V8.75H5v-1.5h2.25V5h1.5v2.25H11v1.5H8.75V11h-1.5z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M8 15c-.099 0-.197-.002-.295-.006A.762.762 0 017.61 15H1.75a.75.75 0 01-.53-1.28l1.328-1.329A7 7 0 118 15zM2.5 8a5.5 5.5 0 115.156 5.49.75.75 0 00-.18.01H3.56l.55-.55a.75.75 0 000-1.06A5.48 5.48 0 012.5 8z",
      fill: "currentColor"
    })]
  });
}
function SpeechBubblePlusIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgSpeechBubblePlusIconV2 : SvgSpeechBubblePlusIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgStarFillIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M12 17.27L18.18 21l-1.64-7.03L22 9.24l-7.19-.61L12 2 9.19 8.63 2 9.24l5.46 4.73L5.82 21 12 17.27z",
      fill: "currentColor"
    })
  });
}
function SvgStarFillIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M7.995 0a.75.75 0 01.714.518l1.459 4.492h4.723a.75.75 0 01.44 1.356l-3.82 2.776 1.459 4.492a.75.75 0 01-1.154.838l-3.82-2.776-3.821 2.776a.75.75 0 01-1.154-.838L4.48 9.142.66 6.366A.75.75 0 011.1 5.01h4.723L7.282.518A.75.75 0 017.995 0z",
      fill: "currentColor"
    })
  });
}
function StarFillIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgStarFillIconV2 : SvgStarFillIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgStarIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M22 9.24l-7.19-.62L12 2 9.19 8.63 2 9.24l5.46 4.73L5.82 21 12 17.27 18.18 21l-1.63-7.03L22 9.24zM12 15.4l-3.76 2.27 1-4.28-3.32-2.88 4.38-.38L12 6.1l1.71 4.04 4.38.38-3.32 2.88 1 4.28L12 15.4z",
      fill: "currentColor"
    })
  });
}
function SvgStarIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M7.995 0a.75.75 0 01.714.518l1.459 4.492h4.723a.75.75 0 01.44 1.356l-3.82 2.776 1.459 4.492a.75.75 0 01-1.154.838l-3.82-2.776-3.821 2.776a.75.75 0 01-1.154-.838L4.48 9.142.66 6.366A.75.75 0 011.1 5.01h4.723L7.282.518A.75.75 0 017.995 0zm0 3.177l-.914 2.814a.75.75 0 01-.713.519h-2.96l2.394 1.739a.75.75 0 01.273.839l-.915 2.814 2.394-1.74a.75.75 0 01.882 0l2.394 1.74-.914-2.814a.75.75 0 01.272-.839l2.394-1.74H9.623a.75.75 0 01-.713-.518l-.915-2.814z",
      fill: "currentColor"
    })
  });
}
function StarIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgStarIconV2 : SvgStarIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgStopCircleFillIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M21 12a9 9 0 11-18 0 9 9 0 0118 0zM10 8a2 2 0 00-2 2v4a2 2 0 002 2h4a2 2 0 002-2v-4a2 2 0 00-2-2h-4z",
      fill: "currentColor",
      stroke: "currentColor",
      strokeWidth: 2
    })
  });
}
function SvgStopCircleFillIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M8 16A8 8 0 108 0a8 8 0 000 16zM6.125 5.5a.625.625 0 00-.625.625v3.75c0 .345.28.625.625.625h3.75c.345 0 .625-.28.625-.625v-3.75a.625.625 0 00-.625-.625h-3.75z",
      fill: "currentColor"
    })
  });
}
function StopCircleFillIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgStopCircleFillIconV2 : SvgStopCircleFillIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgStopCircleIconV1(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M11 9h3.002A1 1 0 0115 10v4a1 1 0 01-1 1h-3.998A1 1 0 019 14v-4a1 1 0 011-1h1z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M12 20a8 8 0 100-16 8 8 0 000 16zm0 2c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z",
      fill: "currentColor"
    })]
  });
}
function SvgStopCircleIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M1.5 8a6.5 6.5 0 1113 0 6.5 6.5 0 01-13 0zM8 0a8 8 0 100 16A8 8 0 008 0zM5.5 6a.5.5 0 01.5-.5h4a.5.5 0 01.5.5v4a.5.5 0 01-.5.5H6a.5.5 0 01-.5-.5V6z",
      fill: "currentColor"
    })
  });
}
function StopCircleIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgStopCircleIconV2 : SvgStopCircleIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgStopIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M8 6h8c1.1 0 2 .9 2 2v8c0 1.1-.9 2-2 2H8c-1.1 0-2-.9-2-2V8c0-1.1.9-2 2-2z",
      fill: "currentColor"
    })
  });
}
function SvgStopIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M4.5 4a.5.5 0 00-.5.5v7a.5.5 0 00.5.5h7a.5.5 0 00.5-.5v-7a.5.5 0 00-.5-.5h-7z",
      fill: "currentColor"
    })
  });
}
function StopIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgStopIconV2 : SvgStopIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgStreamIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M21.984 12.565c.01-.187.016-.375.016-.565 0-5.523-4.477-10-10-10S2 6.477 2 12s4.477 10 10 10c4.572 0 8.428-3.069 9.62-7.258a9.975 9.975 0 00.364-2.177zm-17.83 1.003a8.003 8.003 0 0015.74-.26l-1.773-1.772a3 3 0 00-4.242 0l-2.343 2.343a5 5 0 01-7.072 0l-.31-.311zm-.048-2.876a8.002 8.002 0 0115.74-.26l-.31-.31a5 5 0 00-7.072 0l-2.343 2.342a3 3 0 01-4.242 0l-1.773-1.772z",
      fill: "currentColor"
    })
  });
}
function SvgStreamIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M8 0a8 8 0 100 16A8 8 0 008 0zM1.52 7.48a6.5 6.5 0 0112.722-1.298l-.09-.091a3.75 3.75 0 00-5.304 0L6.091 8.848a2.25 2.25 0 01-3.182 0L1.53 7.47l-.01.01zm.238 2.338A6.5 6.5 0 0014.48 8.52l-.01.01-1.379-1.378a2.25 2.25 0 00-3.182 0L7.152 9.909a3.75 3.75 0 01-5.304 0l-.09-.09z",
      fill: "currentColor"
    })
  });
}
function StreamIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgStreamIconV2 : SvgStreamIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgSyncIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M12 6c3.31 0 6 2.69 6 6h-1.79a.5.5 0 00-.36.85l2.79 2.79c.2.2.51.2.71 0l2.79-2.79c.32-.31.1-.85-.35-.85H20c0-4.42-3.58-8-8-8-1.04 0-2.04.2-2.95.57-.67.27-.85 1.13-.34 1.64.27.27.68.38 1.04.23C10.44 6.15 11.21 6 12 6zm-3.86 5.14L5.35 8.35a.492.492 0 00-.7.01l-2.79 2.79c-.32.31-.1.85.35.85H4c0 4.42 3.58 8 8 8 1.04 0 2.04-.2 2.95-.57.67-.27.85-1.13.34-1.64-.27-.27-.68-.38-1.04-.23-.69.29-1.46.44-2.25.44-3.31 0-6-2.69-6-6h1.79c.45 0 .67-.54.35-.86z",
      fill: "currentColor"
    })
  });
}
function SvgSyncIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M8 2.5a5.48 5.48 0 013.817 1.54l.009.009.5.451H11V6h4V2h-1.5v1.539l-.651-.588A7 7 0 001 8h1.5A5.5 5.5 0 018 2.5zM1 10h4v1.5H3.674l.5.451.01.01A5.5 5.5 0 0013.5 8h1.499a7 7 0 01-11.849 5.048L2.5 12.46V14H1v-4z",
      fill: "currentColor"
    })
  });
}
function SyncIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgSyncIconV2 : SvgSyncIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgTableIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M4 2a2 2 0 00-2 2v17a2 2 0 002 2h16a2 2 0 002-2V4a2 2 0 00-2-2H4zm0 6h4v3H4V8zm6 3V8h4v3h-4zm4 2h-4v3h4v-3zm2 3v-3h4v3h-4zm-2 2h-4v3h4v-3zm2 3v-3h4v3h-4zm0-10V8h4v3h-4zM4 13h4v3H4v-3zm0 5h4v3H4v-3z",
      fill: "currentColor"
    })
  });
}
function SvgTableIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M1 1.75A.75.75 0 011.75 1h12.5a.75.75 0 01.75.75v12.5a.75.75 0 01-.75.75H1.75a.75.75 0 01-.75-.75V1.75zm1.5.75v3h11v-3h-11zm0 11V7H5v6.5H2.5zm4 0h3V7h-3v6.5zM11 7v6.5h2.5V7H11z",
      fill: "currentColor"
    })
  });
}
function TableIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgTableIconV2 : SvgTableIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgTrashIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M15.5 4H18c.55 0 1 .45 1 1s-.45 1-1 1H6c-.55 0-1-.45-1-1s.45-1 1-1h2.5l.71-.71c.18-.18.44-.29.7-.29h4.18c.26 0 .52.11.7.29l.71.71zM8 21c-1.1 0-2-.9-2-2V9c0-1.1.9-2 2-2h8c1.1 0 2 .9 2 2v10c0 1.1-.9 2-2 2H8z",
      fill: "currentColor"
    })
  });
}
function SvgTrashIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M6 0a.75.75 0 00-.712.513L4.46 3H1v1.5h1.077l1.177 10.831A.75.75 0 004 16h8a.75.75 0 00.746-.669L13.923 4.5H15V3h-3.46L10.712.513A.75.75 0 0010 0H6zm3.96 3l-.5-1.5H6.54L6.04 3h3.92zM3.585 4.5l1.087 10h6.654l1.087-10H3.586z",
      fill: "currentColor"
    })
  });
}
function TrashIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgTrashIconV2 : SvgTrashIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgTreeIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M8.5 4.5A3.5 3.5 0 1113 7.855V11.5h5.632c1.334 0 2.368 1.113 2.368 2.412v2.233A3.502 3.502 0 0120 23a3.5 3.5 0 01-1-6.855v-2.233c0-.254-.197-.412-.368-.412H13v2.645A3.502 3.502 0 0112 23a3.5 3.5 0 01-1-6.855V13.5H5.368c-.17 0-.368.158-.368.412v2.233A3.502 3.502 0 014 23a3.5 3.5 0 01-1-6.855v-2.233c0-1.3 1.034-2.412 2.368-2.412H11V7.855A3.502 3.502 0 018.5 4.5zM12 3a1.5 1.5 0 100 3 1.5 1.5 0 000-3zM2.5 19.5a1.5 1.5 0 113 0 1.5 1.5 0 01-3 0zm8 0a1.5 1.5 0 113 0 1.5 1.5 0 01-3 0zm8 0a1.5 1.5 0 113 0 1.5 1.5 0 01-3 0z",
      fill: "currentColor"
    })
  });
}
function SvgTreeIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M2.004 9.602a2.751 2.751 0 103.371 3.47 2.751 2.751 0 005.25 0 2.751 2.751 0 103.371-3.47A2.75 2.75 0 0011.25 7h-2.5v-.604a2.751 2.751 0 10-1.5 0V7h-2.5a2.75 2.75 0 00-2.746 2.602zM2.75 11a1.25 1.25 0 100 2.5 1.25 1.25 0 000-2.5zm4.5-2.5h-2.5a1.25 1.25 0 00-1.242 1.106 2.756 2.756 0 011.867 1.822A2.756 2.756 0 017.25 9.604V8.5zm1.5 0v1.104c.892.252 1.6.942 1.875 1.824a2.756 2.756 0 011.867-1.822A1.25 1.25 0 0011.25 8.5h-2.5zM12 12.25a1.25 1.25 0 112.5 0 1.25 1.25 0 01-2.5 0zm-5.25 0a1.25 1.25 0 102.5 0 1.25 1.25 0 00-2.5 0zM8 5a1.25 1.25 0 110-2.5A1.25 1.25 0 018 5z",
      fill: "currentColor"
    })
  });
}
function TreeIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgTreeIconV2 : SvgTreeIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgUndoIconV1(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("g", {
      clipPath: "url(#UndoIcon_svg__clip0_13917_34581)",
      children: jsx("path", {
        d: "M2.81 6.5h8.69a3 3 0 010 6H7V14h4.5a4.5 4.5 0 000-9H2.81l2.72-2.72-1.06-1.06-4.53 4.53 4.53 4.53 1.06-1.06L2.81 6.5z",
        fill: "currentColor"
      })
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "UndoIcon_svg__clip0_13917_34581",
        children: jsx("path", {
          fill: "#fff",
          transform: "rotate(-180 8 8)",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
function SvgUndoIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("g", {
      clipPath: "url(#UndoIcon_svg__clip0_13917_34581)",
      children: jsx("path", {
        d: "M2.81 6.5h8.69a3 3 0 010 6H7V14h4.5a4.5 4.5 0 000-9H2.81l2.72-2.72-1.06-1.06-4.53 4.53 4.53 4.53 1.06-1.06L2.81 6.5z",
        fill: "currentColor"
      })
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "UndoIcon_svg__clip0_13917_34581",
        children: jsx("path", {
          fill: "#fff",
          transform: "rotate(-180 8 8)",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
function UndoIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgUndoIconV2 : SvgUndoIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgUploadIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M14 16.296h-4c-.55 0-1-.45-1-1v-5H7.41c-.89 0-1.33-1.08-.7-1.71l4.59-4.59a.996.996 0 011.41 0l4.59 4.59c.63.63.18 1.71-.71 1.71H15v5c0 .55-.45 1-1 1zm4 2H6c-.55 0-1 .45-1 1s.45 1 1 1h12c.55 0 1-.45 1-1s-.45-1-1-1z",
      fill: "currentColor"
    })
  });
}
function SvgUploadIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M1 13.56h14v1.5H1v-1.5zM12.53 5.53l-1.06 1.061-2.72-2.72v7.19h-1.5V3.87l-2.72 2.72-1.06-1.06L8 1l4.53 4.53z",
      fill: "currentColor"
    })
  });
}
function UploadIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgUploadIconV2 : SvgUploadIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgUsbIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M18.095 7.236h-2c-.55 0-1 .45-1 1v2c0 .55.45 1 1 1v2h-3v-8h1a.5.5 0 00.4-.8l-2-2.67c-.2-.27-.6-.27-.8 0l-2 2.67a.5.5 0 00.4.8h1v8h-3v-2.07c.83-.44 1.38-1.36 1.14-2.43-.17-.77-.77-1.4-1.52-1.61-1.47-.41-2.82.7-2.82 2.11 0 .85.5 1.56 1.2 1.93v2.07c0 1.1.9 2 2 2h3v3.05c-.86.45-1.39 1.42-1.13 2.49a2.204 2.204 0 004.34-.54c0-.85-.49-1.58-1.2-1.95v-3.05h3c1.1 0 2-.9 2-2v-2c.55 0 1-.45 1-1v-2c-.01-.55-.46-1-1.01-1z",
      fill: "currentColor"
    })
  });
}
function SvgUsbIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M8 0a.75.75 0 01.65.375l1.299 2.25a.75.75 0 01-.65 1.125H8.75V9.5h2.75V8h-.25a.75.75 0 01-.75-.75v-2a.75.75 0 01.75-.75h2a.75.75 0 01.75.75v2a.75.75 0 01-.75.75H13v2.25a.75.75 0 01-.75.75h-3.5v1.668a1.75 1.75 0 11-1.5 0V11h-3.5a.75.75 0 01-.75-.75V7.832a1.75 1.75 0 111.5 0V9.5h2.75V3.75h-.549a.75.75 0 01-.65-1.125l1.3-2.25A.75.75 0 018 0z",
      fill: "currentColor"
    })
  });
}
function UsbIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgUsbIconV2 : SvgUsbIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgUserBadgeIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M14.82 4H19c1.1 0 2 .9 2 2v14c0 1.1-.9 2-2 2H5c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2h4.18C9.6 2.84 10.7 2 12 2c1.3 0 2.4.84 2.82 2zM13 5c0-.55-.45-1-1-1s-1 .45-1 1 .45 1 1 1 1-.45 1-1zm-1 3c1.66 0 3 1.34 3 3s-1.34 3-3 3-3-1.34-3-3 1.34-3 3-3zM6 18.6V20h12v-1.4c0-2-4-3.1-6-3.1s-6 1.1-6 3.1z",
      fill: "currentColor"
    })
  });
}
function SvgUserBadgeIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M8 5.25a2.75 2.75 0 100 5.5 2.75 2.75 0 000-5.5zM6.75 8a1.25 1.25 0 112.5 0 1.25 1.25 0 01-2.5 0z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M4.401 2.5l.386-.867A2.75 2.75 0 017.3 0h1.4a2.75 2.75 0 012.513 1.633l.386.867h1.651a.75.75 0 01.75.75v12a.75.75 0 01-.75.75H2.75a.75.75 0 01-.75-.75v-12a.75.75 0 01.75-.75h1.651zm1.756-.258A1.25 1.25 0 017.3 1.5h1.4c.494 0 .942.29 1.143.742l.114.258H6.043l.114-.258zM8 12a8.71 8.71 0 00-4.5 1.244V4h9v9.244A8.71 8.71 0 008 12zm0 1.5c1.342 0 2.599.364 3.677 1H4.323A7.216 7.216 0 018 13.5z",
      fill: "currentColor"
    })]
  });
}
function UserBadgeIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgUserBadgeIconV2 : SvgUserBadgeIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgUserCircleIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 3c1.66 0 3 1.34 3 3s-1.34 3-3 3-3-1.34-3-3 1.34-3 3-3zM6 15.98a7.2 7.2 0 0012 0c-.03-1.99-4.01-3.08-6-3.08-2 0-5.97 1.09-6 3.08z",
      fill: "currentColor"
    })
  });
}
function SvgUserCircleIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M5.25 6.75a2.75 2.75 0 115.5 0 2.75 2.75 0 01-5.5 0zM8 5.5A1.25 1.25 0 108 8a1.25 1.25 0 000-2.5z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M0 8a8 8 0 1116 0A8 8 0 010 8zm8-6.5a6.5 6.5 0 00-4.773 10.912A8.728 8.728 0 018 11c1.76 0 3.4.52 4.773 1.412A6.5 6.5 0 008 1.5zm3.568 11.934A7.231 7.231 0 008 12.5a7.23 7.23 0 00-3.568.934A6.47 6.47 0 008 14.5a6.47 6.47 0 003.568-1.066z",
      fill: "currentColor"
    })]
  });
}
function UserCircleIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgUserCircleIconV2 : SvgUserCircleIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgUserGroupIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M10.99 8c0 1.66-1.33 3-2.99 3-1.66 0-3-1.34-3-3s1.34-3 3-3 2.99 1.34 2.99 3zm8 0c0 1.66-1.33 3-2.99 3-1.66 0-3-1.34-3-3s1.34-3 3-3 2.99 1.34 2.99 3zM8 13c-2.33 0-7 1.17-7 3.5V18c0 .55.45 1 1 1h12c.55 0 1-.45 1-1v-1.5c0-2.33-4.67-3.5-7-3.5zm7.03.05c.35-.03.68-.05.97-.05 2.33 0 7 1.17 7 3.5V18c0 .55-.45 1-1 1h-5.18c.11-.31.18-.65.18-1v-1.5c0-1.47-.79-2.58-1.93-3.41a.12.12 0 01-.01-.011.092.092 0 00-.03-.029z",
      fill: "currentColor"
    })
  });
}
function SvgUserGroupIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M2.25 3.75a2.75 2.75 0 115.5 0 2.75 2.75 0 01-5.5 0zM5 2.5A1.25 1.25 0 105 5a1.25 1.25 0 000-2.5zM9.502 14H.75a.75.75 0 01-.75-.75V11a.75.75 0 01.164-.469C1.298 9.114 3.077 8 5.125 8c1.76 0 3.32.822 4.443 1.952A5.545 5.545 0 0111.75 9.5c1.642 0 3.094.745 4.041 1.73a.75.75 0 01.209.52v1.5a.75.75 0 01-.75.75H9.502zM1.5 12.5v-1.228C2.414 10.228 3.72 9.5 5.125 9.5c1.406 0 2.71.728 3.625 1.772V12.5H1.5zm8.75 0h4.25v-.432A4.168 4.168 0 0011.75 11c-.53 0-1.037.108-1.5.293V12.5zM11.75 3.5a2.25 2.25 0 100 4.5 2.25 2.25 0 000-4.5zM11 5.75a.75.75 0 111.5 0 .75.75 0 01-1.5 0z",
      fill: "currentColor"
    })
  });
}
function UserGroupIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgUserGroupIconV2 : SvgUserGroupIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgUserIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M16 8c0 2.21-1.79 4-4 4s-4-1.79-4-4 1.79-4 4-4 4 1.79 4 4zM4 18c0-2.66 5.33-4 8-4s8 1.34 8 4v1c0 .55-.45 1-1 1H5c-.55 0-1-.45-1-1v-1z",
      fill: "currentColor"
    })
  });
}
function SvgUserIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M8 1a3.25 3.25 0 100 6.5A3.25 3.25 0 008 1zM6.25 4.25a1.75 1.75 0 113.5 0 1.75 1.75 0 01-3.5 0zM8 9a8.735 8.735 0 00-6.836 3.287.75.75 0 00-.164.469v1.494c0 .414.336.75.75.75h12.5a.75.75 0 00.75-.75v-1.494a.75.75 0 00-.164-.469A8.735 8.735 0 008 9zm-5.5 4.5v-.474A7.232 7.232 0 018 10.5c2.2 0 4.17.978 5.5 2.526v.474h-11z",
      fill: "currentColor"
    })
  });
}
function UserIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgUserIconV2 : SvgUserIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgVisibleIconV1(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsxs("g", {
      clipPath: "url(#VisibleIcon_svg__clip0_12466_34543)",
      fillRule: "evenodd",
      clipRule: "evenodd",
      fill: "currentColor",
      children: [jsx("path", {
        d: "M8 5a3 3 0 100 6 3 3 0 000-6zM6.5 8a1.5 1.5 0 113 0 1.5 1.5 0 01-3 0z"
      }), jsx("path", {
        d: "M8 2A8.389 8.389 0 00.028 7.777a.75.75 0 000 .466 8.389 8.389 0 0015.944 0 .749.749 0 000-.466A8.389 8.389 0 008 2zm0 10.52a6.888 6.888 0 01-6.465-4.51 6.888 6.888 0 0112.93 0A6.888 6.888 0 018 12.52z"
      })]
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "VisibleIcon_svg__clip0_12466_34543",
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
function SvgVisibleIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsxs("g", {
      clipPath: "url(#VisibleIcon_svg__clip0_13123_35205)",
      fillRule: "evenodd",
      clipRule: "evenodd",
      fill: "currentColor",
      children: [jsx("path", {
        d: "M8 5a3 3 0 100 6 3 3 0 000-6zM6.5 8a1.5 1.5 0 113 0 1.5 1.5 0 01-3 0z"
      }), jsx("path", {
        d: "M8 2A8.389 8.389 0 00.028 7.777a.75.75 0 000 .466 8.389 8.389 0 0015.944 0 .75.75 0 000-.466A8.389 8.389 0 008 2zm0 10.52a6.888 6.888 0 01-6.465-4.51 6.888 6.888 0 0112.93 0A6.888 6.888 0 018 12.52z"
      })]
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "VisibleIcon_svg__clip0_13123_35205",
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
function VisibleIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgVisibleIconV2 : SvgVisibleIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgVisibleOffIconV1(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsxs("g", {
      clipPath: "url(#VisibleOffIcon_svg__clip0_12466_34549)",
      fill: "currentColor",
      children: [jsx("path", {
        fillRule: "evenodd",
        clipRule: "evenodd",
        d: "M11.634 13.195l1.336 1.335 1.06-1.06-11.5-11.5-1.06 1.06 1.027 1.028a8.395 8.395 0 00-2.469 3.72.75.75 0 000 .465 8.389 8.389 0 0011.606 4.952zm-1.14-1.14l-1.301-1.301a3 3 0 01-3.946-3.946L3.56 5.121A6.898 6.898 0 001.535 8.01a6.888 6.888 0 008.96 4.045z"
      }), jsx("path", {
        d: "M15.972 8.244a8.384 8.384 0 01-1.945 3.222l-1.061-1.06a6.886 6.886 0 001.499-2.396 6.888 6.888 0 00-8.187-4.293L5.082 2.522a8.389 8.389 0 0110.89 5.256c.05.15.05.314 0 .466z"
      }), jsx("path", {
        d: "M11 8c0 .14-.01.277-.028.411L7.59 5.028A3 3 0 0111 8z"
      })]
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "VisibleOffIcon_svg__clip0_12466_34549",
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
function SvgVisibleOffIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsxs("g", {
      clipPath: "url(#VisibleOffIcon_svg__clip0_13123_35207)",
      fill: "currentColor",
      children: [jsx("path", {
        fillRule: "evenodd",
        clipRule: "evenodd",
        d: "M11.634 13.194l1.335 1.336 1.061-1.06-11.5-11.5-1.06 1.06 1.027 1.028a8.395 8.395 0 00-2.469 3.72.75.75 0 000 .465 8.389 8.389 0 0011.606 4.951zm-1.14-1.139l-1.301-1.301a3 3 0 01-3.946-3.946L3.56 5.121A6.898 6.898 0 001.535 8.01a6.888 6.888 0 008.96 4.045z"
      }), jsx("path", {
        d: "M15.972 8.243a8.384 8.384 0 01-1.946 3.223l-1.06-1.06a6.887 6.887 0 001.499-2.396 6.888 6.888 0 00-8.187-4.293L5.082 2.522a8.389 8.389 0 0110.89 5.256.75.75 0 010 .465z"
      }), jsx("path", {
        d: "M11 8c0 .14-.01.277-.028.411L7.589 5.028A3 3 0 0111 8z"
      })]
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "VisibleOffIcon_svg__clip0_13123_35207",
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
function VisibleOffIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgVisibleOffIconV2 : SvgVisibleOffIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgWarningFillIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M19.53 20.504c1.54 0 2.5-1.67 1.73-3l-7.53-13.01c-.77-1.33-2.69-1.33-3.46 0l-7.53 13.01c-.77 1.33.19 3 1.73 3h15.06zm-7.53-7c-.55 0-1-.45-1-1v-2c0-.55.45-1 1-1s1 .45 1 1v2c0 .55-.45 1-1 1zm-1 2v2h2v-2h-2z",
      fill: "currentColor"
    })
  });
}
function SvgWarningFillIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M8.649 1.374a.75.75 0 00-1.298 0l-7.25 12.5A.75.75 0 00.75 15h14.5a.75.75 0 00.649-1.126l-7.25-12.5zM7.25 10V6.5h1.5V10h-1.5zm1.5 1.75a.75.75 0 11-1.5 0 .75.75 0 011.5 0z",
      fill: "currentColor"
    })
  });
}
function WarningFillIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgWarningFillIconV2 : SvgWarningFillIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgWarningIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M4.47 20.504c-1.54 0-2.5-1.67-1.73-3l7.53-13.01c.77-1.33 2.69-1.33 3.46 0l7.53 13.01c.77 1.33-.19 3-1.73 3H4.47zm15.06-2L12 5.494l-7.53 13.01h15.06zm-8.53-8v2c0 .55.45 1 1 1s1-.45 1-1v-2c0-.55-.45-1-1-1s-1 .45-1 1zm2 7v-2h-2v2h2z",
      fill: "currentColor"
    })
  });
}
function SvgWarningIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M7.25 10V6.5h1.5V10h-1.5zM8 12.5A.75.75 0 108 11a.75.75 0 000 1.5z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M8 1a.75.75 0 01.649.374l7.25 12.5A.75.75 0 0115.25 15H.75a.75.75 0 01-.649-1.126l7.25-12.5A.75.75 0 018 1zm0 2.245L2.052 13.5h11.896L8 3.245z",
      fill: "currentColor"
    })]
  });
}
function WarningIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgWarningIconV2 : SvgWarningIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgXCircleFillIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M2 12C2 6.48 6.48 2 12 2s10 4.48 10 10-4.48 10-10 10S2 17.52 2 12zm11.414 0l2.122-2.121a1.003 1.003 0 000-1.415 1.003 1.003 0 00-1.415 0L12 10.586 9.879 8.464a1.003 1.003 0 00-1.415 0 1.003 1.003 0 000 1.415L10.586 12l-2.122 2.121a1.003 1.003 0 000 1.415 1.003 1.003 0 001.415 0L12 13.414l2.121 2.122a1.003 1.003 0 001.415 0 1.003 1.003 0 000-1.415L13.414 12z",
      fill: "currentColor"
    })
  });
}
function SvgXCircleFillIconV2(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M8 16A8 8 0 108 0a8 8 0 000 16zm1.97-4.97L8 9.06l-1.97 1.97-1.06-1.06L6.94 8 4.97 6.03l1.06-1.06L8 6.94l1.97-1.97 1.06 1.06L9.06 8l1.97 1.97-1.06 1.06z",
      fill: "currentColor"
    })
  });
}
function XCircleFillIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgXCircleFillIconV2 : SvgXCircleFillIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgXCircleIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zM8.464 8.464a1.003 1.003 0 000 1.415L10.586 12l-2.122 2.121a1.003 1.003 0 000 1.415 1.003 1.003 0 001.415 0L12 13.414l2.121 2.122a1.003 1.003 0 001.415 0 1.003 1.003 0 000-1.415L13.414 12l2.122-2.121a1.003 1.003 0 000-1.415 1.003 1.003 0 00-1.415 0L12 10.586 9.879 8.464a1.003 1.003 0 00-1.415 0zM4 12c0 4.41 3.59 8 8 8s8-3.59 8-8-3.59-8-8-8-8 3.59-8 8z",
      fill: "currentColor"
    })
  });
}
function SvgXCircleIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M6.94 8L4.97 6.03l1.06-1.06L8 6.94l1.97-1.97 1.06 1.06L9.06 8l1.97 1.97-1.06 1.06L8 9.06l-1.97 1.97-1.06-1.06L6.94 8z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M0 8a8 8 0 1116 0A8 8 0 010 8zm8-6.5a6.5 6.5 0 100 13 6.5 6.5 0 000-13z",
      fill: "currentColor"
    })]
  });
}
function XCircleIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgXCircleIconV2 : SvgXCircleIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgZoomInIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M15.97 14.472h-.79l-.28-.27a6.5 6.5 0 001.48-5.34c-.47-2.78-2.79-5-5.59-5.34-4.23-.52-7.78 3.04-7.27 7.27.34 2.8 2.56 5.12 5.34 5.59a6.5 6.5 0 005.34-1.48l.27.28v.79l4.26 4.25c.41.41 1.07.41 1.48 0l.01-.01c.41-.41.41-1.07 0-1.48l-4.25-4.26zm-6 0c-2.49 0-4.5-2.01-4.5-4.5s2.01-4.5 4.5-4.5 4.5 2.01 4.5 4.5-2.01 4.5-4.5 4.5zm-.5-6.5c0-.28.22-.5.5-.5s.5.22.5.5v1.5h1.5c.28 0 .5.22.5.5s-.22.5-.5.5h-1.5v1.5c0 .28-.22.5-.5.5s-.5-.22-.5-.5v-1.5h-1.5c-.28 0-.5-.22-.5-.5s.22-.5.5-.5h1.5v-1.5z",
      fill: "currentColor"
    })
  });
}
function SvgZoomInIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 17",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M8.75 7.25H11v1.5H8.75V11h-1.5V8.75H5v-1.5h2.25V5h1.5v2.25z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M8 1a7 7 0 104.39 12.453l2.55 2.55 1.06-1.06-2.55-2.55A7 7 0 008 1zM2.5 8a5.5 5.5 0 1111 0 5.5 5.5 0 01-11 0z",
      fill: "currentColor"
    })]
  });
}
function ZoomInIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgZoomInIconV2 : SvgZoomInIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgZoomOutIconV1(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 24 24",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M15.972 14.472h-.79l-.28-.27a6.5 6.5 0 001.48-5.34c-.47-2.78-2.79-5-5.59-5.34a6.505 6.505 0 00-7.27 7.27c.34 2.8 2.56 5.12 5.34 5.59a6.5 6.5 0 005.34-1.48l.27.28v.79l4.26 4.25c.41.41 1.07.41 1.48 0l.01-.01c.41-.41.41-1.07 0-1.48l-4.25-4.26zm-6 0c-2.49 0-4.5-2.01-4.5-4.5s2.01-4.5 4.5-4.5 4.5 2.01 4.5 4.5-2.01 4.5-4.5 4.5zm2.5-4.5c0-.28-.22-.5-.5-.5h-4c-.28 0-.5.22-.5.5s.22.5.5.5h4c.28 0 .5-.22.5-.5z",
      fill: "currentColor"
    })
  });
}
function SvgZoomOutIconV2(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 17",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M11 7.25H5v1.5h6v-1.5z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M1 8a7 7 0 1112.45 4.392l2.55 2.55-1.06 1.061-2.55-2.55A7 7 0 011 8zm7-5.5a5.5 5.5 0 100 11 5.5 5.5 0 000-11z",
      fill: "currentColor"
    })]
  });
}
function ZoomOutIcon(props) {
  const {
    USE_NEW_ICONS
  } = useDesignSystemFlags();
  const component = USE_NEW_ICONS ? SvgZoomOutIconV2 : SvgZoomOutIconV1;
  return jsx(Icon, {
    ...props,
    component: component
  });
}

function SvgAppIcon(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M2.75 1a1.75 1.75 0 100 3.5 1.75 1.75 0 000-3.5zM8 1a1.75 1.75 0 100 3.5A1.75 1.75 0 008 1zm5.25 0a1.75 1.75 0 100 3.5 1.75 1.75 0 000-3.5zM2.75 6.25a1.75 1.75 0 100 3.5 1.75 1.75 0 000-3.5zm5.25 0a1.75 1.75 0 100 3.5 1.75 1.75 0 000-3.5zm5.25 0a1.75 1.75 0 100 3.5 1.75 1.75 0 000-3.5zM2.75 11.5a1.75 1.75 0 100 3.5 1.75 1.75 0 000-3.5zm5.25 0A1.75 1.75 0 108 15a1.75 1.75 0 000-3.5zm5.25 0a1.75 1.75 0 100 3.5 1.75 1.75 0 000-3.5z",
      fill: "currentColor"
    })
  });
}
function AppIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgAppIcon
  });
}

function SvgBinaryIcon(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M1 3a2 2 0 114 0v2a2 2 0 11-4 0V3zm2-.5a.5.5 0 00-.5.5v2a.5.5 0 001 0V3a.5.5 0 00-.5-.5zm3.378-.628c.482 0 .872-.39.872-.872h1.5v4.25H10v1.5H6v-1.5h1.25V3.206c-.27.107-.564.166-.872.166H6v-1.5h.378zm5 0c.482 0 .872-.39.872-.872h1.5v4.25H15v1.5h-4v-1.5h1.25V3.206c-.27.107-.564.166-.872.166H11v-1.5h.378zM6 11a2 2 0 114 0v2a2 2 0 11-4 0v-2zm2-.5a.5.5 0 00-.5.5v2a.5.5 0 001 0v-2a.5.5 0 00-.5-.5zm-6.622-.378c.482 0 .872-.39.872-.872h1.5v4.25H5V15H1v-1.5h1.25v-2.044c-.27.107-.564.166-.872.166H1v-1.5h.378zm10 0c.482 0 .872-.39.872-.872h1.5v4.25H15V15h-4v-1.5h1.25v-2.044c-.27.107-.564.166-.872.166H11v-1.5h.378z",
      fill: "currentColor"
    })
  });
}
function BinaryIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgBinaryIcon
  });
}

function SvgBoldIcon(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M4.75 3a.75.75 0 00-.75.75v8.5c0 .414.336.75.75.75h4.375a2.875 2.875 0 001.496-5.33A2.875 2.875 0 008.375 3H4.75zm.75 5.75v2.75h3.625a1.375 1.375 0 000-2.75H5.5zm2.877-1.5a1.375 1.375 0 00-.002-2.75H5.5v2.75h2.877z",
      fill: "currentColor"
    })
  });
}
function BoldIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgBoldIcon
  });
}

function SvgBracketsCurlyIcon(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M5.5 2a2.75 2.75 0 00-2.75 2.75v1C2.75 6.44 2.19 7 1.5 7H1v2h.5c.69 0 1.25.56 1.25 1.25v1A2.75 2.75 0 005.5 14H6v-1.5h-.5c-.69 0-1.25-.56-1.25-1.25v-1c0-.93-.462-1.752-1.168-2.25A2.747 2.747 0 004.25 5.75v-1c0-.69.56-1.25 1.25-1.25H6V2h-.5zM13.25 4.75A2.75 2.75 0 0010.5 2H10v1.5h.5c.69 0 1.25.56 1.25 1.25v1c0 .93.462 1.752 1.168 2.25a2.747 2.747 0 00-1.168 2.25v1c0 .69-.56 1.25-1.25 1.25H10V14h.5a2.75 2.75 0 002.75-2.75v-1c0-.69.56-1.25 1.25-1.25h.5V7h-.5c-.69 0-1.25-.56-1.25-1.25v-1z",
      fill: "currentColor"
    })
  });
}
function BracketsCurlyIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgBracketsCurlyIcon
  });
}

function SvgBracketsSquareIcon(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M1 1.75A.75.75 0 011.75 1H5v1.5H2.5v11H5V15H1.75a.75.75 0 01-.75-.75V1.75zm12.5.75H11V1h3.25a.75.75 0 01.75.75v12.5a.75.75 0 01-.75.75H11v-1.5h2.5v-11z",
      fill: "currentColor"
    })
  });
}
function BracketsSquareIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgBracketsSquareIcon
  });
}

function SvgBracketsXIcon(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsxs("g", {
      clipPath: "url(#BracketsXIcon_svg__clip0_14891_29556)",
      fill: "currentColor",
      children: [jsx("path", {
        d: "M1.75 4.75A2.75 2.75 0 014.5 2H5v1.5h-.5c-.69 0-1.25.56-1.25 1.25v1c0 .93-.462 1.752-1.168 2.25a2.747 2.747 0 011.168 2.25v1c0 .69.56 1.25 1.25 1.25H5V14h-.5a2.75 2.75 0 01-2.75-2.75v-1C1.75 9.56 1.19 9 .5 9H0V7h.5c.69 0 1.25-.56 1.25-1.25v-1zM11.5 2a2.75 2.75 0 012.75 2.75v1c0 .69.56 1.25 1.25 1.25h.5v2h-.5c-.69 0-1.25.56-1.25 1.25v1A2.75 2.75 0 0111.5 14H11v-1.5h.5c.69 0 1.25-.56 1.25-1.25v-1c0-.93.462-1.752 1.168-2.25a2.747 2.747 0 01-1.168-2.25v-1c0-.69-.56-1.25-1.25-1.25H11V2h.5z"
      }), jsx("path", {
        d: "M4.97 6.03L6.94 8 4.97 9.97l1.06 1.06L8 9.06l1.97 1.97 1.06-1.06L9.06 8l1.97-1.97-1.06-1.06L8 6.94 6.03 4.97 4.97 6.03z"
      })]
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "BracketsXIcon_svg__clip0_14891_29556",
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
function BracketsXIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgBracketsXIcon
  });
}

function SvgCatalogIcon(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M4.75 0A2.75 2.75 0 002 2.75V13.5A2.5 2.5 0 004.5 16h8.75a.75.75 0 00.75-.75V.75a.75.75 0 00-.75-.75h-8.5zm7.75 11V1.5H4.75c-.69 0-1.25.56-1.25 1.25v8.458a2.492 2.492 0 011-.208h8zm-9 2.5a1 1 0 001 1h8v-2h-8a1 1 0 00-1 1z",
      fill: "currentColor"
    })
  });
}
function CatalogIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgCatalogIcon
  });
}

function SvgChartLineIcon(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M1 1v13.25c0 .414.336.75.75.75H15v-1.5H2.5V1H1z",
      fill: "currentColor"
    }), jsx("path", {
      d: "M15.03 5.03l-1.06-1.06L9.5 8.44 7 5.94 3.47 9.47l1.06 1.06L7 8.06l2.5 2.5 5.53-5.53z",
      fill: "currentColor"
    })]
  });
}
function ChartLineIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgChartLineIcon
  });
}

function SvgCheckCircleBadgeIcon(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M10.47 5.47l1.06 1.06L7 11.06 4.47 8.53l1.06-1.06L7 8.94l3.47-3.47zM16 12.5a3.5 3.5 0 11-7 0 3.5 3.5 0 017 0z",
      fill: "currentColor"
    }), jsx("path", {
      d: "M1.5 8a6.5 6.5 0 0113-.084c.54.236 1.031.565 1.452.967a8 8 0 10-7.07 7.07 5.008 5.008 0 01-.966-1.454A6.5 6.5 0 011.5 8z",
      fill: "currentColor"
    })]
  });
}
function CheckCircleBadgeIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgCheckCircleBadgeIcon
  });
}

function SvgCloudKeyIcon(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M3.394 5.586a4.752 4.752 0 019.351.946A3.754 3.754 0 0115.787 9H14.12a2.248 2.248 0 00-1.871-1H12a.75.75 0 01-.75-.75v-.5a3.25 3.25 0 00-6.476-.402.75.75 0 01-.697.657A2.75 2.75 0 004 12.49V14a.75.75 0 01-.179-.021 4.25 4.25 0 01-.427-8.393zM15.25 10.5h-4.291a3 3 0 10-.13 1.5H12v2h1.5v-2h1v2H16v-2.75a.75.75 0 00-.75-.75zM8 9.5a1.5 1.5 0 100 3 1.5 1.5 0 000-3z",
      fill: "currentColor"
    })
  });
}
function CloudKeyIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgCloudKeyIcon
  });
}

function SvgCloudModelIcon(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M3.394 5.586a4.752 4.752 0 019.351.946A3.754 3.754 0 0115.787 9H14.12a2.248 2.248 0 00-1.871-1H12a.75.75 0 01-.75-.75v-.5a3.25 3.25 0 00-6.476-.402.75.75 0 01-.697.657A2.75 2.75 0 004 12.49V14a.75.75 0 01-.179-.021 4.25 4.25 0 01-.427-8.393z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M8 7a2.25 2.25 0 012.03 3.22l.5.5a2.25 2.25 0 11-1.06 1.06l-.5-.5A2.25 2.25 0 118 7zm.75 2.25a.75.75 0 10-1.5 0 .75.75 0 001.5 0zm3.5 3.5a.75.75 0 10-1.5 0 .75.75 0 001.5 0z",
      fill: "currentColor"
    })]
  });
}
function CloudModelIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgCloudModelIcon
  });
}

function SvgColorFillIcon(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M7.5 1v1.59l4.88 4.88a.75.75 0 010 1.06l-4.242 4.243a2.75 2.75 0 01-3.89 0l-2.421-2.422a2.75 2.75 0 010-3.889L6 2.29V1h1.5zM6 8V4.41L2.888 7.524a1.25 1.25 0 000 1.768l2.421 2.421a1.25 1.25 0 001.768 0L10.789 8 7.5 4.71V8H6zm7.27 1.51a.76.76 0 00-1.092.001 8.53 8.53 0 00-1.216 1.636c-.236.428-.46.953-.51 1.501-.054.576.083 1.197.587 1.701a2.385 2.385 0 003.372 0c.505-.504.644-1.126.59-1.703-.05-.55-.274-1.075-.511-1.503a8.482 8.482 0 00-1.22-1.633zm-.995 2.363c.138-.25.3-.487.451-.689.152.201.313.437.452.687.19.342.306.657.33.913.02.228-.03.377-.158.505a.885.885 0 01-1.25 0c-.125-.125-.176-.272-.155-.501.024-.256.14-.572.33-.915z",
      fill: "currentColor"
    })
  });
}
function ColorFillIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgColorFillIcon
  });
}

function SvgConnectIcon(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M7.78 3.97L5.03 1.22a.75.75 0 00-1.06 0L1.22 3.97a.75.75 0 000 1.06l2.75 2.75a.75.75 0 001.06 0l2.75-2.75a.75.75 0 000-1.06zm-1.59.53L4.5 6.19 2.81 4.5 4.5 2.81 6.19 4.5zM15 11.75a3.25 3.25 0 10-6.5 0 3.25 3.25 0 006.5 0zM11.75 10a1.75 1.75 0 110 3.5 1.75 1.75 0 010-3.5z",
      fill: "currentColor"
    }), jsx("path", {
      d: "M14.25 1H9v1.5h4.5V7H15V1.75a.75.75 0 00-.75-.75zM1 9v5.25c0 .414.336.75.75.75H7v-1.5H2.5V9H1z",
      fill: "currentColor"
    })]
  });
}
function ConnectIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgConnectIcon
  });
}

function SvgCursorIcon(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("g", {
      clipPath: "url(#CursorIcon_svg__clip0_14914_35410)",
      children: jsx("path", {
        fillRule: "evenodd",
        clipRule: "evenodd",
        d: "M1.22 1.22a.75.75 0 01.802-.169l13.5 5.25a.75.75 0 01-.043 1.413L9.597 9.597l-1.883 5.882a.75.75 0 01-1.413.043l-5.25-13.5a.75.75 0 01.169-.802zm1.847 1.847l3.864 9.937 1.355-4.233a.75.75 0 01.485-.485l4.233-1.355-9.937-3.864z",
        fill: "currentColor"
      })
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "CursorIcon_svg__clip0_14914_35410",
        children: jsx("path", {
          fill: "#fff",
          transform: "matrix(-1 0 0 1 16 0)",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
function CursorIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgCursorIcon
  });
}

function SvgDagIcon(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M8 1.75A.75.75 0 018.75 1h5.5a.75.75 0 01.75.75v3.5a.75.75 0 01-.75.75h-5.5A.75.75 0 018 5.25v-1H5.5c-.69 0-1.25.56-1.25 1.25h2a.75.75 0 01.75.75v3.5a.75.75 0 01-.75.75h-2c0 .69.56 1.25 1.25 1.25H8v-1a.75.75 0 01.75-.75h5.5a.75.75 0 01.75.75v3.5a.75.75 0 01-.75.75h-5.5a.75.75 0 01-.75-.75v-1H5.5a2.75 2.75 0 01-2.75-2.75h-2A.75.75 0 010 9.75v-3.5a.75.75 0 01.75-.75h2A2.75 2.75 0 015.5 2.75H8v-1zm1.5.75v2h4v-2h-4zM1.5 9V7h4v2h-4zm8 4.5v-2h4v2h-4z",
      fill: "currentColor"
    })
  });
}
function DagIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgDagIcon
  });
}

function SvgDecimalIcon(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M3 10a3 3 0 106 0V6a3 3 0 00-6 0v4zm3 1.5A1.5 1.5 0 014.5 10V6a1.5 1.5 0 113 0v4A1.5 1.5 0 016 11.5zM10 10a3 3 0 106 0V6a3 3 0 10-6 0v4zm3 1.5a1.5 1.5 0 01-1.5-1.5V6a1.5 1.5 0 013 0v4a1.5 1.5 0 01-1.5 1.5z",
      fill: "currentColor"
    }), jsx("path", {
      d: "M1 13a1 1 0 100-2 1 1 0 000 2z",
      fill: "currentColor"
    })]
  });
}
function DecimalIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgDecimalIcon
  });
}

function SvgDotsCircleIcon(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsxs("g", {
      clipPath: "url(#DotsCircleIcon_svg__clip0_14891_29603)",
      fill: "currentColor",
      children: [jsx("path", {
        d: "M6 8a.75.75 0 11-1.5 0A.75.75 0 016 8zM8 8.75a.75.75 0 100-1.5.75.75 0 000 1.5zM10.75 8.75a.75.75 0 100-1.5.75.75 0 000 1.5z"
      }), jsx("path", {
        fillRule: "evenodd",
        clipRule: "evenodd",
        d: "M8 0a8 8 0 100 16A8 8 0 008 0zM1.5 8a6.5 6.5 0 1113 0 6.5 6.5 0 01-13 0z"
      })]
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "DotsCircleIcon_svg__clip0_14891_29603",
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
function DotsCircleIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgDotsCircleIcon
  });
}

function SvgFontIcon(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("g", {
      clipPath: "url(#FontIcon_svg__clip0_13123_35195)",
      children: jsx("path", {
        fillRule: "evenodd",
        clipRule: "evenodd",
        d: "M5.197 3.473a.75.75 0 00-1.393-.001L-.006 13H1.61l.6-1.5h4.562l.596 1.5h1.614L5.197 3.473zM6.176 10L4.498 5.776 2.809 10h3.367zm4.07-2.385c.593-.205 1.173-.365 1.754-.365a1.5 1.5 0 011.42 1.014A3.764 3.764 0 0012 8c-.741 0-1.47.191-2.035.607A2.301 2.301 0 009 10.5c0 .81.381 1.464.965 1.893.565.416 1.294.607 2.035.607.524 0 1.042-.096 1.5-.298V13H15V8.75a3 3 0 00-3-3c-.84 0-1.614.23-2.245.448l.49 1.417zM13.5 10.5a.804.804 0 00-.353-.685C12.897 9.631 12.5 9.5 12 9.5c-.5 0-.897.131-1.146.315a.804.804 0 00-.354.685c0 .295.123.515.354.685.25.184.645.315 1.146.315.502 0 .897-.131 1.147-.315.23-.17.353-.39.353-.685z",
        fill: "currentColor"
      })
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "FontIcon_svg__clip0_13123_35195",
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
function FontIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgFontIcon
  });
}

function SvgFunctionIcon(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("g", {
      clipPath: "url(#FunctionIcon_svg__clip0_16055_28727)",
      children: jsx("path", {
        fillRule: "evenodd",
        clipRule: "evenodd",
        d: "M9.93 2.988c-.774-.904-2.252-.492-2.448.682L7.094 6h2.005a2.75 2.75 0 012.585 1.81l.073.202 2.234-2.063 1.018 1.102-2.696 2.489.413 1.137c.18.494.65.823 1.175.823H15V13h-1.1a2.75 2.75 0 01-2.585-1.81l-.198-.547-2.61 2.408-1.017-1.102 3.07-2.834-.287-.792A1.25 1.25 0 009.099 7.5H6.844l-.846 5.076c-.405 2.43-3.464 3.283-5.067 1.412l1.139-.976c.774.904 2.252.492 2.448-.682l.805-4.83H3V6h2.573l.43-2.576C6.407.994 9.465.14 11.068 2.012l-1.138.976z",
        fill: "currentColor"
      })
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "FunctionIcon_svg__clip0_16055_28727",
        children: jsx("path", {
          fill: "#fff",
          transform: "matrix(-1 0 0 1 16 0)",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
function FunctionIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgFunctionIcon
  });
}

function SvgGiftIcon(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M3 3.25A2.25 2.25 0 015.25 1C6.365 1 7.36 1.522 8 2.335A3.494 3.494 0 0110.75 1a2.25 2.25 0 012.122 3h1.378a.75.75 0 01.75.75v3a.75.75 0 01-.75.75H14v5.75a.75.75 0 01-.75.75H2.75a.75.75 0 01-.75-.75V8.5h-.25A.75.75 0 011 7.75v-3A.75.75 0 011.75 4h1.378A2.246 2.246 0 013 3.25zM5.25 4h1.937A2 2 0 005.25 2.5a.75.75 0 000 1.5zm2 1.5H2.5V7h4.75V5.5zm0 3H3.5v5h3.75v-5zm1.5 5v-5h3.75v5H8.75zm0-6.5V5.5h4.75V7H8.75zm.063-3h1.937a.75.75 0 000-1.5A2 2 0 008.813 4z",
      fill: "currentColor"
    })
  });
}
function GiftIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgGiftIcon
  });
}

function SvgItalicIcon(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M9.648 4.5H12V3H6v1.5h2.102l-1.75 7H4V13h6v-1.5H7.898l1.75-7z",
      fill: "currentColor"
    })
  });
}
function ItalicIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgItalicIcon
  });
}

function SvgLettersIcon(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M6.25 1h2.174a2.126 2.126 0 011.81 3.243 2.126 2.126 0 01-1.36 3.761H6.25a.75.75 0 01-.75-.75V1.75A.75.75 0 016.25 1zM7 6.504V5.252h1.874a.626.626 0 110 1.252H7zm2.05-3.378c0 .345-.28.625-.625.626H7.001L7 2.5h1.424c.346 0 .626.28.626.626zM3.307 6a.75.75 0 01.697.473L6.596 13H4.982l-.238-.6H1.855l-.24.6H0l2.61-6.528A.75.75 0 013.307 6zm-.003 2.776l.844 2.124H2.455l.85-2.124z",
      fill: "currentColor"
    }), jsx("path", {
      d: "M12.5 15a2.5 2.5 0 002.5-2.5h-1.5a1 1 0 11-2 0v-1.947c0-.582.472-1.053 1.053-1.053.523 0 .947.424.947.947v.053H15v-.053A2.447 2.447 0 0012.553 8 2.553 2.553 0 0010 10.553V12.5a2.5 2.5 0 002.5 2.5z",
      fill: "currentColor"
    })]
  });
}
function LettersIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgLettersIcon
  });
}

function SvgMenuIcon(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M15 4H1V2.5h14V4zm0 4.75H1v-1.5h14v1.5zm0 4.75H1V12h14v1.5z",
      fill: "currentColor"
    })
  });
}
function MenuIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgMenuIcon
  });
}

function SvgNotificationOffIcon(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M14.47 13.53l-12-12-1 1L3.28 4.342A4.992 4.992 0 003 6v1.99c0 .674-.2 1.332-.573 1.892l-1.301 1.952A.75.75 0 001.75 13h3.5v.25a2.75 2.75 0 105.5 0V13h1.19l1.53 1.53 1-1zM13.038 8.5A3.409 3.409 0 0113 7.99V6a5 5 0 00-7.965-4.026l1.078 1.078A3.5 3.5 0 0111.5 6v1.99c0 .158.008.316.023.472l.038.038h1.477zM4.5 6c0-.14.008-.279.024-.415L10.44 11.5H3.151l.524-.786A4.91 4.91 0 004.5 7.99V6zm2.25 7.25V13h2.5v.25a1.25 1.25 0 11-2.5 0z",
      fill: "currentColor"
    })
  });
}
function NotificationOffIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgNotificationOffIcon
  });
}

function SvgNumbersIcon(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M7.889 1A2.389 2.389 0 005.5 3.389H7c0-.491.398-.889.889-.889h.371a.74.74 0 01.292 1.42l-1.43.613A2.675 2.675 0 005.5 6.992V8h5V6.5H7.108c.12-.26.331-.472.604-.588l1.43-.613A2.24 2.24 0 008.26 1H7.89zM2.75 6a1.5 1.5 0 01-1.5 1.5H1V9h.25c.546 0 1.059-.146 1.5-.401V11.5H1V13h5v-1.5H4.25V6h-1.5zM10 12.85A2.15 2.15 0 0012.15 15h.725a2.125 2.125 0 001.617-3.504 2.138 2.138 0 00-1.656-3.521l-.713.008A2.15 2.15 0 0010 10.133v.284h1.5v-.284a.65.65 0 01.642-.65l.712-.009a.638.638 0 11.008 1.276H12v1.5h.875a.625.625 0 110 1.25h-.725a.65.65 0 01-.65-.65v-.267H10v.267z",
      fill: "currentColor"
    })
  });
}
function NumbersIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgNumbersIcon
  });
}

function SvgPipelineIcon(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M10.75 6.75A5.75 5.75 0 005 1H1.75a.75.75 0 00-.75.75V6c0 .414.336.75.75.75H5a.25.25 0 01.25.25v2.25A5.75 5.75 0 0011 15h3.25a.75.75 0 00.75-.75V10a.75.75 0 00-.75-.75H11a.25.25 0 01-.25-.25V6.75zM5.5 2.53a4.25 4.25 0 013.75 4.22V9a1.75 1.75 0 001.25 1.678v2.793A4.25 4.25 0 016.75 9.25V7A1.75 1.75 0 005.5 5.322V2.53zM4 2.5v2.75H2.5V2.5H4zm9.5 8.25H12v2.75h1.5v-2.75z",
      fill: "currentColor"
    })
  });
}
function PipelineIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgPipelineIcon
  });
}

function SvgPlugIcon(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M14.168 2.953l.893-.892L14 1l-.893.893a4.001 4.001 0 00-5.077.48l-.884.884a.75.75 0 000 1.061l4.597 4.596a.75.75 0 001.06 0l.884-.884a4.001 4.001 0 00.48-5.077zM12.627 6.97l-.354.353-3.536-3.535.354-.354a2.5 2.5 0 113.536 3.536zM7.323 10.152L5.91 8.737l1.414-1.414-1.06-1.06-1.415 1.414-.53-.53a.75.75 0 00-1.06 0l-.885.883a4.001 4.001 0 00-.48 5.077L1 14l1.06 1.06.893-.892a4.001 4.001 0 005.077-.48l.884-.885a.75.75 0 000-1.06l-.53-.53 1.414-1.415-1.06-1.06-1.415 1.414zm-3.889 2.475a2.5 2.5 0 003.536 0l.353-.354-3.535-3.536-.354.354a2.5 2.5 0 000 3.536z",
      fill: "currentColor"
    })
  });
}
function PlugIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgPlugIcon
  });
}

function SvgSaveIcon(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M10 9.25H6v1.5h4v-1.5z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M1 1.75A.75.75 0 011.75 1H11a.75.75 0 01.53.22l3.25 3.25c.141.14.22.331.22.53v9.25a.75.75 0 01-.75.75H1.75a.75.75 0 01-.75-.75V1.75zm1.5.75H5v3.75c0 .414.336.75.75.75h4.5a.75.75 0 00.75-.75V2.81l2.5 2.5v8.19h-11v-11zm4 0h3v3h-3v-3z",
      fill: "currentColor"
    })]
  });
}
function SaveIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgSaveIcon
  });
}

function SvgSlidersIcon(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M2 3.104V2h1.5v1.104a2.751 2.751 0 010 5.292V14H2V8.396a2.751 2.751 0 010-5.292zM1.5 5.75a1.25 1.25 0 112.5 0 1.25 1.25 0 01-2.5 0zM12.5 2v1.104a2.751 2.751 0 000 5.292V14H14V8.396a2.751 2.751 0 000-5.292V2h-1.5zm.75 2.5a1.25 1.25 0 100 2.5 1.25 1.25 0 000-2.5zM7.25 14v-1.104a2.751 2.751 0 010-5.292V2h1.5v5.604a2.751 2.751 0 010 5.292V14h-1.5zM8 11.5A1.25 1.25 0 118 9a1.25 1.25 0 010 2.5z",
      fill: "currentColor"
    })
  });
}
function SlidersIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgSlidersIcon
  });
}

function SvgSortAscendingIcon(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M11.5.94l4.03 4.03-1.06 1.06-2.22-2.22V10h-1.5V3.81L8.53 6.03 7.47 4.97 11.5.94zM1 4.5h4V6H1V4.5zM1 12.5h10V14H1v-1.5zM8 8.5H1V10h7V8.5z",
      fill: "currentColor"
    })
  });
}
function SortAscendingIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgSortAscendingIcon
  });
}

function SvgSortDescendingIcon(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M1 3.5h10V2H1v1.5zm0 8h4V10H1v1.5zm7-4H1V6h7v1.5zm3.5 7.56l4.03-4.03-1.06-1.06-2.22 2.22V6h-1.5v6.19L8.53 9.97l-1.06 1.06 4.03 4.03z",
      fill: "currentColor"
    })
  });
}
function SortDescendingIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgSortDescendingIcon
  });
}

function SvgSortUnsortedIcon(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M11.5.94L7.47 4.97l1.06 1.06 2.22-2.22v8.38L8.53 9.97l-1.06 1.06 4.03 4.03 4.03-4.03-1.06-1.06-2.22 2.22V3.81l2.22 2.22 1.06-1.06L11.5.94zM6 3.5H1V5h5V3.5zM6 11.5H1V13h5v-1.5zM1 7.5h5V9H1V7.5z",
      fill: "currentColor"
    })
  });
}
function SortUnsortedIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgSortUnsortedIcon
  });
}

function SvgStorefrontIcon(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M3.52 2.3a.75.75 0 01.6-.3h7.76a.75.75 0 01.6.3l2.37 3.158a.75.75 0 01.15.45v.842c0 .04-.003.077-.009.115A2.311 2.311 0 0114 8.567v5.683a.75.75 0 01-.75.75H2.75a.75.75 0 01-.75-.75V8.567A2.311 2.311 0 011 6.75v-.841a.75.75 0 01.15-.45l2.37-3.16zm7.605 6.068c.368.337.847.557 1.375.6V13.5h-9V8.968a2.303 2.303 0 001.375-.6c.411.377.96.607 1.563.607.602 0 1.15-.23 1.562-.607.411.377.96.607 1.563.607.602 0 1.15-.23 1.562-.607zm2.375-2.21v.532l-.001.019a.813.813 0 01-1.623 0 .754.754 0 00-.008-.076.756.756 0 00.012-.133V4L13.5 6.16zm-3.113.445a.762.762 0 00-.013.106.813.813 0 01-1.624-.019V3.5h1.63v3c0 .035.002.07.007.103zM7.25 3.5v3.19a.813.813 0 01-1.624.019.757.757 0 00-.006-.064V3.5h1.63zM4.12 4L2.5 6.16v.531l.001.019a.813.813 0 001.619.045V4z",
      fill: "currentColor"
    })
  });
}
function StorefrontIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgStorefrontIcon
  });
}

function SvgTextBoxIcon(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M1.75 1a.75.75 0 00-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 00.75-.75V1.75a.75.75 0 00-.75-.75H1.75zm.75 12.5v-11h11v11h-11zM5 6h2.25v5.5h1.5V6H11V4.5H5V6z",
      fill: "currentColor"
    })
  });
}
function TextBoxIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgTextBoxIcon
  });
}

function SvgUnderlineIcon(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M4.544 6.466L4.6 2.988l1.5.024-.056 3.478A1.978 1.978 0 1010 6.522V3h1.5v3.522a3.478 3.478 0 11-6.956-.056zM12 13H4v-1.5h8V13z",
      fill: "currentColor"
    })
  });
}
function UnderlineIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgUnderlineIcon
  });
}

function SvgVariableIcon(props) {
  return jsx("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: jsx("path", {
      d: "M3.73 13.2h.4v-1.109h-.257c-.78 0-1.084-.345-1.084-1.228v-1.74c0-.858-.507-1.347-1.472-1.448v-.173c.965-.108 1.472-.596 1.472-1.455v-1.71c0-.883.304-1.228 1.084-1.228h.257V2h-.4c-1.638 0-2.353.662-2.353 2.158v1.46c0 .906-.37 1.234-1.377 1.234v1.466c1.013.006 1.377.334 1.377 1.234v1.472c0 1.502.715 2.176 2.353 2.176zM4.994 11.036H6.59L7.92 8.771h.1l1.336 2.265h1.668L8.897 7.74 11 4.521H9.374l-1.288 2.26h-.1L6.691 4.52H4.976l2.127 3.285-2.11 3.23zM12.27 13.2c1.638 0 2.354-.674 2.354-2.176V9.552c0-.9.363-1.228 1.376-1.234V6.852c-1.007 0-1.377-.328-1.377-1.234v-1.46C14.623 2.662 13.909 2 12.27 2h-.4v1.109h.257c.786 0 1.09.345 1.09 1.228v1.71c0 .859.5 1.347 1.466 1.455v.173c-.965.1-1.466.59-1.466 1.448v1.74c0 .883-.31 1.228-1.09 1.228h-.257V13.2h.4z",
      fill: "currentColor"
    })
  });
}
function VariableIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgVariableIcon
  });
}

function SvgWorkspacesIcon(props) {
  return jsxs("svg", {
    width: "1em",
    height: "1em",
    viewBox: "0 0 16 16",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("path", {
      d: "M2.5 1a.75.75 0 00-.75.75v3c0 .414.336.75.75.75H6V4H3.25V2.5h9.5V4H10v1.5h3.5a.75.75 0 00.75-.75v-3A.75.75 0 0013.5 1h-11z",
      fill: "currentColor"
    }), jsx("path", {
      fillRule: "evenodd",
      clipRule: "evenodd",
      d: "M0 12.25c0-1.26.848-2.322 2.004-2.648A2.75 2.75 0 014.75 7h2.5V4h1.5v3h2.5a2.75 2.75 0 012.746 2.602 2.751 2.751 0 11-3.371 3.47 2.751 2.751 0 01-5.25 0A2.751 2.751 0 010 12.25zM2.75 11a1.25 1.25 0 100 2.5 1.25 1.25 0 000-2.5zm2.625.428a2.756 2.756 0 00-1.867-1.822A1.25 1.25 0 014.75 8.5h2.5v1.104c-.892.252-1.6.942-1.875 1.824zM8.75 9.604V8.5h2.5c.642 0 1.17.483 1.242 1.106a2.756 2.756 0 00-1.867 1.822A2.756 2.756 0 008.75 9.604zM12 12.25a1.25 1.25 0 112.5 0 1.25 1.25 0 01-2.5 0zm-5.25 0a1.25 1.25 0 102.5 0 1.25 1.25 0 00-2.5 0z",
      fill: "currentColor"
    })]
  });
}
function WorkspacesIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgWorkspacesIcon
  });
}

function getAccordionEmotionStyles(clsPrefix, theme) {
  const classItem = `.${clsPrefix}-item`;
  const classItemActive = `${classItem}-active`;
  const classHeader = `.${clsPrefix}-header`;
  const classContent = `.${clsPrefix}-content`;
  const classContentBox = `.${clsPrefix}-content-box`;
  const classArrow = `.${clsPrefix}-arrow`;
  const styles = {
    border: '0 none',
    background: 'none',
    [classItem]: {
      border: '0 none',
      [`&:hover`]: {
        [classHeader]: {
          color: theme.colors.actionPrimaryBackgroundHover
        },
        [classArrow]: {
          color: theme.colors.actionPrimaryBackgroundHover
        }
      },
      [`&:active`]: {
        [classHeader]: {
          color: theme.colors.actionPrimaryBackgroundPress
        },
        [classArrow]: {
          color: theme.colors.actionPrimaryBackgroundPress
        }
      }
    },
    [classHeader]: {
      color: theme.colors.textPrimary,
      fontWeight: 600,
      '&:focus-visible': {
        outlineColor: `${theme.colors.primary} !important`,
        outlineStyle: 'auto !important'
      }
    },
    [`& > ${classItem} > ${classHeader} > ${classArrow}`]: {
      fontSize: theme.general.iconSize,
      right: 12
    },
    [classArrow]: {
      color: theme.colors.textSecondary
    },
    [`& > ${classItemActive} > ${classHeader} > ${classArrow}`]: {
      transform: 'translateY(-50%) rotate(180deg)'
    },
    [classContent]: {
      border: '0 none',
      backgroundColor: theme.colors.backgroundPrimary
    },
    [classContentBox]: {
      padding: '8px 16px 16px'
    },
    [`& > ${classItem} > ${classHeader}`]: {
      padding: '6px 44px 6px 0',
      lineHeight: theme.typography.lineHeightBase
    },
    ...getAnimationCss(theme.options.enableAnimation)
  };
  return /*#__PURE__*/css(styles, process.env.NODE_ENV === "production" ? "" : ";label:getAccordionEmotionStyles;");
}
const AccordionPanel = _ref => {
  let {
    dangerouslySetAntdProps,
    dangerouslyAppendEmotionCSS,
    children,
    ...props
  } = _ref;
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Collapse.Panel, {
      ...props,
      ...dangerouslySetAntdProps,
      css: dangerouslyAppendEmotionCSS,
      children: jsx(RestoreAntDDefaultClsPrefix, {
        children: children
      })
    })
  });
};
const Accordion = /* #__PURE__ */(() => {
  const Accordion = _ref2 => {
    let {
      dangerouslySetAntdProps,
      dangerouslyAppendEmotionCSS,
      displayMode = 'multiple',
      ...props
    } = _ref2;
    const {
      theme,
      getPrefixedClassName
    } = useDesignSystemTheme();
    // While this component is called `Accordion` for correctness, in AntD it is called `Collapse`.
    const clsPrefix = getPrefixedClassName('collapse');
    return jsx(DesignSystemAntDConfigProvider, {
      children: jsx(Collapse, {
        expandIcon: () => jsx(ChevronDownIcon, {}),
        expandIconPosition: "right",
        accordion: displayMode === 'single',
        ...props,
        ...dangerouslySetAntdProps,
        css: [getAccordionEmotionStyles(clsPrefix, theme), dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:Accordion;"]
      })
    });
  };
  Accordion.Panel = AccordionPanel;
  return Accordion;
})();

// TODO: Replace with custom icons
// TODO: Reuse in Alert
const filledIconsMap = {
  error: DangerFillIcon,
  warning: WarningFillIcon,
  success: CheckCircleFillIcon,
  info: InfoFillIcon
};
function SeverityIcon(props) {
  const FilledIcon = filledIconsMap[props.severity];
  return jsx(FilledIcon, {
    ...props
  });
}

const Alert = _ref => {
  let {
    dangerouslySetAntdProps,
    closable = true,
    ...props
  } = _ref;
  const {
    theme,
    getPrefixedClassName
  } = useDesignSystemTheme();
  const clsPrefix = getPrefixedClassName('alert');
  const mergedProps = {
    ...props,
    type: props.type || 'error',
    showIcon: true,
    closable
  };
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Alert$1, {
      ...mergedProps,
      className: classnames(mergedProps.className),
      css: getAlertEmotionStyles(clsPrefix, theme, mergedProps),
      icon: jsx(SeverityIcon, {
        severity: mergedProps.type
      })
      // Antd calls this prop `closeText` but we can use it to set any React element to replace the close icon.
      ,
      closeText: mergedProps.closable && jsx(CloseIcon, {
        css: /*#__PURE__*/css({
          fontSize: theme.general.iconSize
        }, process.env.NODE_ENV === "production" ? "" : ";label:Alert;")
      })
      // Always set a description for consistent styling (e.g. icon size)
      ,
      description: props.description || ' ',
      ...dangerouslySetAntdProps
    })
  });
};
const getAlertEmotionStyles = (clsPrefix, theme, props) => {
  const classCloseIcon = `.${clsPrefix}-close-icon`;
  const classCloseButton = `.${clsPrefix}-close-button`;
  const classCloseText = `.${clsPrefix}-close-text`;
  const classDescription = `.${clsPrefix}-description`;
  const classMessage = `.${clsPrefix}-message`;
  const classWithDescription = `.${clsPrefix}-with-description`;
  const classWithIcon = `.${clsPrefix}-icon`;
  const ALERT_ICON_HEIGHT = 16;
  const ALERT_ICON_FONT_SIZE = 16;
  const styles = {
    // General
    padding: theme.spacing.sm,
    [`${classMessage}, &${classWithDescription} ${classMessage}`]: {
      // TODO(giles): These three rules are all the same as the H3 styles. We can refactor them out into a shared object.
      fontSize: theme.typography.fontSizeBase,
      fontWeight: theme.typography.typographyBoldFontWeight,
      lineHeight: theme.typography.lineHeightBase
    },
    [`${classDescription}`]: {
      lineHeight: theme.typography.lineHeightBase
    },
    // Icons
    [classCloseButton]: {
      fontSize: ALERT_ICON_FONT_SIZE,
      marginRight: 12
    },
    [classCloseIcon]: {
      '&:focus-visible': {
        outlineStyle: 'auto',
        outlineColor: theme.colors.primary
      }
    },
    [`${classCloseIcon}, ${classCloseButton}`]: {
      lineHeight: theme.typography.lineHeightBase,
      height: ALERT_ICON_HEIGHT,
      width: ALERT_ICON_HEIGHT,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center'
    },
    [classWithIcon]: {
      fontSize: ALERT_ICON_FONT_SIZE,
      marginTop: 2
    },
    [`${classCloseIcon}, ${classCloseButton}, ${classCloseText} > span`]: {
      lineHeight: theme.typography.lineHeightBase,
      height: ALERT_ICON_HEIGHT,
      width: ALERT_ICON_HEIGHT,
      fontSize: ALERT_ICON_FONT_SIZE,
      marginTop: 2,
      '& > span': {
        lineHeight: theme.typography.lineHeightBase
      }
    },
    // No description
    ...(!props.description && {
      [classMessage]: {
        margin: 0
      },
      [classDescription]: {
        display: 'none'
      }
    }),
    // Warning
    ...(props.type === 'warning' && {
      color: theme.colors.textValidationWarning,
      borderColor: theme.colors.yellow300
    }),
    // Error
    ...(props.type === 'error' && {
      color: theme.colors.textValidationDanger,
      borderColor: theme.colors.red300
    }),
    // Banner
    ...(props.banner && {
      borderStyle: 'solid',
      borderWidth: `${theme.general.borderWidth}px 0`
    }),
    // After closed
    '&[data-show="false"]': {
      borderWidth: 0,
      padding: 0,
      width: 0,
      height: 0
    },
    ...getAnimationCss(theme.options.enableAnimation)
  };
  return /*#__PURE__*/css(styles, process.env.NODE_ENV === "production" ? "" : ";label:getAlertEmotionStyles;");
};

const AutoComplete = /* #__PURE__ */(() => {
  const AutoComplete = _ref => {
    let {
      dangerouslySetAntdProps,
      ...props
    } = _ref;
    const {
      theme
    } = useDesignSystemTheme();
    return jsx(DesignSystemAntDConfigProvider, {
      children: jsx(AutoComplete$1, {
        dropdownStyle: {
          boxShadow: theme.general.shadowLow
        },
        ...props,
        ...dangerouslySetAntdProps,
        css: /*#__PURE__*/css(getAnimationCss(theme.options.enableAnimation), process.env.NODE_ENV === "production" ? "" : ";label:AutoComplete;")
      })
    });
  };
  AutoComplete.Option = AutoComplete$1.Option;
  return AutoComplete;
})();

const Breadcrumb = /* #__PURE__ */(() => {
  const Breadcrumb = _ref => {
    let {
      dangerouslySetAntdProps,
      includeTrailingCaret = true,
      ...props
    } = _ref;
    const {
      theme,
      classNamePrefix
    } = useDesignSystemTheme();
    const {
      USE_NEW_ICONS
    } = useDesignSystemFlags();
    const separatorClass = `.${classNamePrefix}-breadcrumb-separator`;
    const styles = /*#__PURE__*/css({
      // `antd` forces the last anchor to be black, so that it doesn't look like an anchor
      // (even though it is one). This undoes that; if the user wants to make the last
      // text-colored, they can do that by not using an anchor.
      'span:last-child a': {
        color: theme.colors.primary,
        // TODO: Need to pull a global color for anchor hover/focus. Discuss with Ginny.
        ':hover, :focus': {
          color: '#2272B4'
        }
      },
      // TODO: Consider making this global within dubois components
      a: {
        '&:focus-visible': {
          outlineColor: `${theme.colors.primary} !important`,
          outlineStyle: 'auto !important'
        }
      },
      ...(USE_NEW_ICONS && {
        [separatorClass]: {
          fontSize: theme.general.iconFontSizeNew
        },
        '& > span': {
          display: 'inline-flex',
          alignItems: 'center'
        }
      })
    }, process.env.NODE_ENV === "production" ? "" : ";label:styles;");
    return jsx(DesignSystemAntDConfigProvider, {
      children: jsxs(Breadcrumb$1, {
        separator: jsx(ChevronRightIcon, {}),
        ...props,
        ...dangerouslySetAntdProps,
        css: /*#__PURE__*/css(getAnimationCss(theme.options.enableAnimation), styles, process.env.NODE_ENV === "production" ? "" : ";label:Breadcrumb;"),
        children: [props.children, includeTrailingCaret && props.children && jsx(Breadcrumb.Item, {
          children: " "
        })]
      })
    });
  };
  Breadcrumb.Item = Breadcrumb$1.Item;
  Breadcrumb.Separator = Breadcrumb$1.Separator;
  return Breadcrumb;
})();

const ColorVars = {
  primary: 'textPrimary',
  secondary: 'textSecondary',
  info: 'textValidationInfo',
  error: 'textValidationDanger',
  success: 'textValidationSuccess',
  warning: 'textValidationWarning'
};

/**
 * Recursively appends `!important` to all CSS properties in an Emotion `CSSObject`.
 * Used to ensure that we always override Ant styles, without worrying about selector precedence.
 */
function importantify(obj) {
  return _mapValues(obj, (value, key) => {
    if (_isString(value) || _isNumber(value) || _isBoolean(value)) {
      // Make sure we don't double-append important
      if (_isString(value) && _endsWith(value, '!important')) {
        return value;
      }
      if (_isNumber(value)) {
        if (unitless[key]) {
          return `${value}!important`;
        }
        return `${value}px!important`;
      }
      return `${value}!important`;
    }
    if (_isNil(value)) {
      return value;
    }
    return importantify(value);
  });
}

/**
 * Returns a text color, in case of invalid/missing key and missing fallback color it will return textPrimary
 * @param theme
 * @param key - key of TypographyColor
 * @param fallbackColor - color to return as fallback -- used to remove tertiary check inline
 */
function getTypographyColor(theme, key, fallbackColor) {
  if (theme && key && Object(theme.colors).hasOwnProperty(ColorVars[key])) {
    return theme.colors[ColorVars[key]];
  }
  return fallbackColor !== null && fallbackColor !== void 0 ? fallbackColor : theme.colors.textPrimary;
}

/**
 * Returns validation color based on state, has default validation colors if params are not provided
 * @param theme
 * @param validationState
 * @param errorColor
 * @param warningColor
 * @param successColor
 */
function getValidationStateColor(theme, validationState) {
  let {
    errorColor,
    warningColor,
    successColor
  } = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : {};
  switch (validationState) {
    case 'error':
      return errorColor || theme.colors.actionDangerPrimaryBackgroundDefault;
    case 'warning':
      return warningColor || theme.colors.textValidationWarning;
    case 'success':
      return successColor || theme.colors.textValidationSuccess;
    default:
      return undefined;
  }
}

function getDefaultStyles(theme) {
  return {
    backgroundColor: theme.colors.actionDefaultBackgroundDefault,
    borderColor: theme.colors.actionDefaultBorderDefault,
    color: theme.colors.actionDefaultTextDefault,
    lineHeight: theme.typography.lineHeightBase,
    textDecoration: 'none',
    '&:hover': {
      backgroundColor: theme.colors.actionDefaultBackgroundHover,
      borderColor: theme.colors.actionDefaultBorderHover,
      color: theme.colors.actionDefaultTextHover
    },
    '&:active': {
      backgroundColor: theme.colors.actionDefaultBackgroundPress,
      borderColor: theme.colors.actionDefaultBorderPress,
      color: theme.colors.actionDefaultTextPress
    }
  };
}
function getPrimaryStyles(theme) {
  return {
    backgroundColor: theme.colors.actionPrimaryBackgroundDefault,
    borderColor: 'transparent',
    color: theme.colors.actionPrimaryTextDefault,
    textShadow: 'none',
    '&:hover': {
      backgroundColor: theme.colors.actionPrimaryBackgroundHover,
      borderColor: 'transparent',
      color: theme.colors.actionPrimaryTextHover
    },
    '&:active': {
      backgroundColor: theme.colors.actionPrimaryBackgroundPress,
      borderColor: 'transparent',
      color: theme.colors.actionPrimaryTextPress
    }
  };
}
function getLinkStyles$1(theme) {
  return {
    backgroundColor: theme.colors.actionTertiaryBackgroundDefault,
    borderColor: theme.colors.actionTertiaryBackgroundDefault,
    color: theme.colors.actionTertiaryTextDefault,
    '&:hover': {
      backgroundColor: theme.colors.actionTertiaryBackgroundHover,
      borderColor: 'transparent',
      color: theme.colors.actionTertiaryTextHover
    },
    '&:active': {
      backgroundColor: theme.colors.actionTertiaryBackgroundPress,
      borderColor: 'transparent',
      color: theme.colors.actionTertiaryTextPress
    },
    '&[disabled]:hover': {
      background: 'none',
      color: theme.colors.actionDisabledText
    }
  };
}
function getPrimaryDangerStyles(theme) {
  return {
    backgroundColor: theme.colors.actionDangerPrimaryBackgroundDefault,
    borderColor: 'transparent',
    color: theme.colors.white,
    '&:hover': {
      backgroundColor: theme.colors.actionDangerPrimaryBackgroundHover,
      borderColor: 'transparent',
      color: theme.colors.white
    },
    '&:active': {
      backgroundColor: theme.colors.actionDangerPrimaryBackgroundPress,
      borderColor: 'transparent',
      color: theme.colors.white
    },
    '&:focus-visible': {
      outlineColor: theme.colors.actionDangerPrimaryBackgroundDefault
    }
  };
}
function getSecondaryDangerStyles(theme) {
  return {
    backgroundColor: theme.colors.actionDangerDefaultBackgroundDefault,
    borderColor: theme.colors.actionDangerDefaultBorderDefault,
    color: theme.colors.actionDangerDefaultTextDefault,
    '&:hover': {
      backgroundColor: theme.colors.actionDangerDefaultBackgroundHover,
      borderColor: theme.colors.actionDangerDefaultBorderHover,
      color: theme.colors.actionDangerDefaultTextHover
    },
    '&:active': {
      backgroundColor: theme.colors.actionDangerDefaultBackgroundPress,
      borderColor: theme.colors.actionDangerDefaultBorderPress,
      color: theme.colors.actionDangerDefaultTextPress
    },
    '&:focus-visible': {
      outlineColor: theme.colors.actionDangerPrimaryBackgroundDefault
    }
  };
}
function getDisabledStyles(theme) {
  return {
    backgroundColor: theme.colors.actionDisabledBackground,
    borderColor: 'transparent',
    color: theme.colors.actionDisabledText,
    '&:hover': {
      backgroundColor: theme.colors.actionDisabledBackground,
      borderColor: 'transparent',
      color: theme.colors.actionDisabledText
    },
    '&:active': {
      backgroundColor: theme.colors.actionDisabledBackground,
      borderColor: 'transparent',
      color: theme.colors.actionDisabledText
    }
  };
}
function getDisabledTertiaryStyles(theme) {
  return {
    backgroundColor: theme.colors.actionTertiaryBackgroundDefault,
    borderColor: 'transparent',
    color: theme.colors.actionDisabledText,
    '&:hover': {
      backgroundColor: theme.colors.actionTertiaryBackgroundDefault,
      borderColor: 'transparent',
      color: theme.colors.actionDisabledText
    },
    '&:active': {
      backgroundColor: theme.colors.actionTertiaryBackgroundDefault,
      borderColor: 'transparent',
      color: theme.colors.actionDisabledText
    }
  };
}

function getEndIconClsName(theme) {
  return `${theme.general.iconfontCssPrefix}-btn-end-icon`;
}
const getButtonEmotionStyles = _ref => {
  let {
    theme,
    classNamePrefix,
    loading,
    withIcon,
    onlyIcon,
    isAnchor,
    enableAnimation,
    size,
    type,
    isFlex,
    useNewIcons,
    useFocusPseudoClass,
    forceIconStyles
  } = _ref;
  const clsIcon = `.${theme.general.iconfontCssPrefix}`;
  const clsEndIcon = `.${getEndIconClsName(theme)}`;
  const clsLoadingIcon = `.${classNamePrefix}-btn-loading-icon`;
  const clsIconOnly = `.${classNamePrefix}-btn-icon-only`;
  const classPrimary = `.${classNamePrefix}-btn-primary`;
  const classLink = `.${classNamePrefix}-btn-link`;
  const classDangerous = `.${classNamePrefix}-btn-dangerous`;
  const SMALL_BUTTON_HEIGHT = 24;
  const tertiaryColors = {
    background: theme.colors.actionTertiaryBackgroundDefault,
    color: theme.colors.actionTertiaryTextDefault,
    '&:hover': {
      background: theme.colors.actionTertiaryBackgroundHover,
      color: theme.colors.actionTertiaryTextHover
    },
    '&:active': {
      background: theme.colors.actionTertiaryBackgroundPress,
      color: theme.colors.actionTertiaryTextPress
    }
  };
  const iconCss = {
    fontSize: useNewIcons ? theme.general.iconFontSizeNew : theme.general.buttonInnerHeight - 4,
    ...(!isFlex && {
      // verticalAlign used by AntD to move icon and label to center
      // TODO(schs): Try to move buttons to flexbox to solve this. Main problem is that flex-inline and inline-block
      //  behave differently (vertically align of multiple buttons is off). See https://codepen.io/qfactor/pen/JXVzBe
      verticalAlign: useNewIcons ? -4 : -5,
      ...(onlyIcon && {
        verticalAlign: -3
      }),
      // verticalAlign used by AntD to move icon and label to center
      // TODO(schs): Try to move buttons to flexbox to solve this. Main problem is that flex-inline and inline-block
      //  behave differently (vertically align of multiple buttons is off). See https://codepen.io/qfactor/pen/JXVzBe
      // Need to make sure not to apply this to regular buttons as it will offset the icons
      ...(!onlyIcon && {
        verticalAlign: useNewIcons ? -3 : -4
      })
    }),
    lineHeight: 0,
    ...(size === 'small' && {
      lineHeight: theme.typography.lineHeightSm,
      ...((onlyIcon || forceIconStyles) && {
        fontSize: 16,
        ...(isFlex && {
          height: 16
        })
      })
    })
  };
  const inactiveIconCss = {
    color: theme.colors.grey600
  };
  const endIconCssSelector = `span > ${clsEndIcon} > ${clsIcon}`;
  const styles = {
    lineHeight: theme.typography.lineHeightBase,
    boxShadow: 'none',
    height: theme.general.heightSm,
    ...(isFlex && {
      display: 'inline-flex',
      alignItems: 'center',
      justifyContent: 'center',
      verticalAlign: 'middle'
    }),
    ...(!onlyIcon && !forceIconStyles && {
      '&&': {
        padding: '4px 12px',
        ...(size === 'small' && {
          padding: '0 8px'
        })
      }
    }),
    ...((onlyIcon || forceIconStyles) && {
      width: theme.general.heightSm
    }),
    ...(size === 'small' && {
      height: SMALL_BUTTON_HEIGHT,
      lineHeight: theme.typography.lineHeightBase,
      ...((onlyIcon || forceIconStyles) && {
        width: SMALL_BUTTON_HEIGHT,
        paddingTop: 0,
        paddingBottom: 0,
        verticalAlign: 'middle'
      })
    }),
    '&:focus-visible': {
      outlineStyle: 'solid',
      outlineWidth: '2px',
      outlineOffset: '1px',
      outlineColor: theme.isDarkMode ? theme.colors.actionDefaultBorderFocus : theme.colors.primary
    },
    ...getDefaultStyles(theme),
    [`&${classPrimary}`]: {
      ...getPrimaryStyles(theme)
    },
    [`&${classLink}`]: {
      ...getLinkStyles$1(theme),
      ...(type === 'link' && {
        padding: 'unset',
        height: 'auto',
        border: 'none',
        boxShadow: 'none',
        '&[disabled],&:hover': {
          background: 'none'
        }
      })
    },
    [`&${classDangerous}${classPrimary}`]: {
      ...getPrimaryDangerStyles(theme)
    },
    [`&${classDangerous}`]: {
      ...getSecondaryDangerStyles(theme)
    },
    [`&[disabled], &${classDangerous}:disabled`]: {
      ...getDisabledStyles(theme),
      ...((onlyIcon || forceIconStyles) && {
        backgroundColor: 'transparent',
        '&:hover': {
          backgroundColor: 'transparent'
        },
        '&:active': {
          backgroundColor: 'transparent'
        }
      })
    },
    // Loading styles
    ...(loading && {
      '::before': {
        opacity: 0
      },
      [`${clsLoadingIcon}`]: {
        ...(onlyIcon ? {
          // In case of only icon, the icon is already centered but vertically not aligned, this fixes that
          verticalAlign: 'middle'
        } : {
          // Position loading indicator in center
          // This would break vertical centering of loading circle when onlyIcon is true
          position: 'absolute'
        }),
        ...(!isFlex && !forceIconStyles && {
          // Normally we'd do `transform: translateX(-50%)` but `antd` crushes that with injected inline `style`.
          left: 'calc(50% - 7px)'
        }),
        // Re-enable animation for the loading spinner, since it can be disabled by the global animation CSS.
        svg: {
          animationDuration: '1s !important'
        }
      },
      [`> :not(${clsLoadingIcon})`]: {
        // Hide all content except loading icon
        opacity: 0,
        visibility: 'hidden',
        // Add horizontal space for icon
        ...(withIcon && {
          paddingLeft: theme.spacing.sm * 2 + 14
        })
      }
    }),
    // Icon styles
    [`> ${clsIcon} + span, > span + ${clsIcon}`]: {
      marginRight: 0
    },
    [`> ${clsIcon}`]: iconCss,
    [`> ${endIconCssSelector}`]: {
      ...iconCss,
      marginLeft: size === 'small' ? 8 : 12
    },
    ...(!type && {
      [`&:enabled:not(:hover):not(:active) > ${clsIcon}`]: inactiveIconCss
    }),
    ...(!type && {
      [`&:enabled:not(:hover):not(:active) > ${endIconCssSelector}`]: inactiveIconCss
    }),
    // Disable animations
    [`&[${classNamePrefix}-click-animating-without-extra-node='true']::after`]: {
      display: 'none'
    },
    [`&${clsIconOnly}`]: {
      border: 'none',
      [`&:enabled:not(${classLink})`]: {
        ...tertiaryColors,
        color: theme.colors.textSecondary,
        '&:hover > .anticon': {
          color: tertiaryColors['&:hover'].color
        },
        '&:active > .anticon': {
          color: tertiaryColors['&:active'].color
        }
      },
      [`&:enabled:not(${classLink}) > .anticon`]: {
        color: theme.colors.textSecondary
      },
      ...(isAnchor && {
        lineHeight: `${theme.general.heightSm}px`,
        ...getLinkStyles$1(theme),
        '&:disabled': {
          color: theme.colors.actionDisabledText
        }
      }),
      '&[disabled]:hover': {
        backgroundColor: 'transparent'
      }
    },
    [`&:focus`]: {
      ...(useFocusPseudoClass && {
        outlineStyle: 'solid',
        outlineWidth: '2px',
        outlineOffset: '1px',
        outlineColor: theme.isDarkMode ? theme.colors.actionDefaultBorderFocus : theme.colors.primary
      }),
      [`${clsLoadingIcon}`]: {
        ...(onlyIcon && {
          // Mitigate wrong left offset for loading state with onlyIcon
          left: 0
        })
      }
    },
    ...(forceIconStyles && {
      padding: '0 6px',
      lineHeight: theme.typography.lineHeightSm,
      color: theme.colors.textSecondary,
      '& > span': {
        verticalAlign: -1,
        height: theme.general.heightSm / 2,
        width: theme.general.heightSm / 2
      },
      [`& > ${clsLoadingIcon} .anticon`]: {
        // left: `calc(50% - 6px)!important`,
        height: theme.general.heightSm / 2,
        width: theme.general.heightSm / 2,
        padding: 0
      }
    }),
    ...getAnimationCss(enableAnimation)
  };

  // Moved outside main style object because state & selector matching in the already existing object keys can create bugs and unwanted overwrites
  const typeStyles = {
    ...(type === 'tertiary' && {
      [`&:enabled:not(${clsIconOnly})`]: tertiaryColors,
      [`&${classLink}[disabled]`]: {
        ...getDisabledTertiaryStyles(theme)
      }
    })
  };
  const importantStyles = importantify(styles);
  const importantTypeStyles = importantify(typeStyles);
  return /*#__PURE__*/css(importantStyles, importantTypeStyles, process.env.NODE_ENV === "production" ? "" : ";label:getButtonEmotionStyles;");
};
const Button = /* #__PURE__ */(() => {
  const Button = /*#__PURE__*/forwardRef(function Button( // Keep size out of props passed to AntD to make deprecation and eventual removal have 0 impact
  _ref2, ref) {
    let {
      dangerouslySetAntdProps,
      children,
      size,
      type,
      endIcon,
      dangerouslySetForceIconStyles,
      dangerouslyUseFocusPseudoClass,
      ...props
    } = _ref2;
    const {
      theme,
      classNamePrefix
    } = useDesignSystemTheme();
    const {
      USE_FLEX_BUTTON: isFlex,
      USE_NEW_ICONS: useNewIcons
    } = useDesignSystemFlags();
    const clsEndIcon = getEndIconClsName(theme);
    return jsx(DesignSystemAntDConfigProvider, {
      children: jsx(Button$1, {
        ...props,
        css: getButtonEmotionStyles({
          theme,
          classNamePrefix,
          loading: Boolean(props.loading),
          withIcon: Boolean(props.icon),
          onlyIcon: Boolean((props.icon || endIcon) && !children),
          isAnchor: Boolean(props.href && !type),
          danger: Boolean(props.danger),
          enableAnimation: theme.options.enableAnimation,
          size: size || 'middle',
          type,
          isFlex,
          useNewIcons,
          forceIconStyles: Boolean(dangerouslySetForceIconStyles),
          useFocusPseudoClass: Boolean(dangerouslyUseFocusPseudoClass)
        }),
        href: props.disabled ? undefined : props.href,
        ...dangerouslySetAntdProps,
        ref: ref,
        type: type === 'tertiary' ? 'link' : type,
        children: children && jsxs("span", {
          style: {
            visibility: props !== null && props !== void 0 && props.loading ? 'hidden' : 'visible',
            ...(isFlex && {
              display: 'inline-flex',
              alignItems: 'center'
            })
          },
          children: [children, endIcon && jsx("span", {
            className: clsEndIcon,
            style: {
              ...(isFlex && {
                display: 'inline-flex',
                alignItems: 'center'
              })
            },
            children: endIcon
          })]
        })
      })
    });
  });

  // This is needed for other Ant components that wrap Button, such as Tooltip, to correctly
  // identify it as an Ant button.
  // This should be removed if the component is rewritten to no longer be a wrapper around Ant.
  // See: https://github.com/ant-design/ant-design/blob/6dd39c1f89b4d6632e6ed022ff1bc275ca1e0f1f/components/button/button.tsx#L291
  Button.__ANT_BUTTON = true;
  return Button;
})();

function getCheckboxEmotionStyles(clsPrefix, theme) {
  let isHorizontal = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : false;
  let useNewCheckboxStyles = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : false;
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
  const styles = {
    ...(useNewCheckboxStyles && {
      [`.${clsPrefix}`]: {
        top: 'unset',
        lineHeight: theme.typography.lineHeightBase
      },
      [`&${classWrapper}, ${classWrapper}`]: {
        alignItems: 'center',
        lineHeight: theme.typography.lineHeightBase
      }
    }),
    // Top level styles are for the unchecked state
    [classInner]: {
      borderColor: theme.colors.actionDefaultBorderDefault
    },
    // Layout styling
    [`&${classContainer}`]: {
      display: 'flex',
      flexDirection: 'column',
      rowGap: theme.spacing.sm,
      columnGap: 0
    },
    ...(isHorizontal && {
      [`&${classContainer}`]: {
        display: 'flex',
        flexDirection: 'row',
        columnGap: theme.spacing.sm,
        rowGap: 0,
        [`& > ${classContainer}-item`]: {
          marginRight: 0
        }
      }
    }),
    // Keyboard focus
    [`${classInput}:focus-visible + ${classInner}`]: {
      outlineWidth: '2px',
      outlineColor: theme.colors.primary,
      outlineOffset: '4px',
      outlineStyle: 'solid'
    },
    // Hover
    [hoverSelector]: {
      backgroundColor: theme.colors.actionDefaultBackgroundHover,
      borderColor: theme.colors.actionPrimaryBackgroundHover
    },
    // Mouse pressed
    [pressSelector]: {
      backgroundColor: theme.colors.actionDefaultBackgroundPress,
      borderColor: theme.colors.actionPrimaryBackgroundPress
    },
    // Checked state
    [classChecked]: {
      '&::after': {
        border: 'none'
      },
      [defaultSelector]: {
        backgroundColor: theme.colors.actionPrimaryBackgroundDefault,
        borderColor: 'transparent'
      },
      // Checked hover
      [hoverSelector]: {
        backgroundColor: theme.colors.actionPrimaryBackgroundHover,
        borderColor: 'transparent'
      },
      // Checked and mouse pressed
      [pressSelector]: {
        backgroundColor: theme.colors.actionPrimaryBackgroundPress,
        borderColor: 'transparent'
      }
    },
    // Indeterminate
    [classIndeterminate]: {
      [classInner]: {
        backgroundColor: theme.colors.actionPrimaryBackgroundDefault,
        borderColor: theme.colors.actionPrimaryBackgroundDefault,
        // The after pseudo-element is used for the check image itself
        '&:after': {
          backgroundColor: theme.colors.white,
          height: '3px',
          width: '8px',
          borderRadius: '4px'
        }
      },
      // Indeterminate hover
      [hoverSelector]: {
        backgroundColor: theme.colors.actionPrimaryBackgroundHover,
        borderColor: 'transparent'
      },
      // Indeterminate and mouse pressed
      [pressSelector]: {
        backgroundColor: theme.colors.actionPrimaryBackgroundPress
      }
    },
    // Disabled state
    [`&${classDisabledWrapper}`]: {
      [classDisabled]: {
        // Disabled Checked
        [`&${classChecked}`]: {
          [classInner]: {
            backgroundColor: theme.colors.actionDisabledBackground,
            borderColor: theme.colors.actionDisabledBackground
          },
          // Disabled checked hover
          [hoverSelector]: {
            backgroundColor: theme.colors.actionDisabledBackground,
            borderColor: theme.colors.actionDisabledBackground
          }
        },
        // Disabled indeterminate
        [`&${classIndeterminate}`]: {
          [classInner]: {
            backgroundColor: theme.colors.actionDisabledBackground,
            borderColor: theme.colors.actionDisabledBackground
          },
          // Disabled indeterminate hover
          [hoverSelector]: {
            backgroundColor: theme.colors.actionDisabledBackground,
            borderColor: theme.colors.actionDisabledBackground
          }
        },
        // Disabled unchecked
        [classInner]: {
          backgroundColor: theme.colors.actionDisabledBackground,
          borderColor: theme.colors.actionDisabledBackground,
          // The after pseudo-element is used for the check image itself
          '&:after': {
            borderColor: theme.colors.white
          }
        },
        // Disabled hover
        [hoverSelector]: {
          backgroundColor: theme.colors.actionDisabledBackground,
          borderColor: theme.colors.actionDisabledBackground
        },
        '& + span': {
          color: theme.colors.actionDisabledText
        }
      }
    },
    // Animation
    ...getAnimationCss(theme.options.enableAnimation)
  };
  return styles;
}
const getWrapperStyle = _ref => {
  let {
    clsPrefix,
    theme,
    wrapperStyle = {},
    useNewStyles
  } = _ref;
  const styles = {
    height: theme.typography.lineHeightBase,
    lineHeight: theme.typography.lineHeightBase,
    ...(useNewStyles && {
      [`&& + .${clsPrefix}-hint, && + .${clsPrefix}-form-message`]: {
        paddingLeft: theme.spacing.lg,
        marginTop: 0
      }
    }),
    ...wrapperStyle
  };
  return /*#__PURE__*/css(styles, process.env.NODE_ENV === "production" ? "" : ";label:getWrapperStyle;");
};
const DuboisCheckbox = /*#__PURE__*/forwardRef(function Checkbox(_ref2, ref) {
  let {
    isChecked,
    onChange,
    children,
    isDisabled = false,
    style,
    wrapperStyle,
    dangerouslySetAntdProps,
    className,
    ...restProps
  } = _ref2;
  const {
    theme,
    classNamePrefix,
    getPrefixedClassName
  } = useDesignSystemTheme();
  const {
    USE_NEW_CHECKBOX_STYLES
  } = useDesignSystemFlags();
  const clsPrefix = getPrefixedClassName('checkbox');
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx("div", {
      className: classnames(className, `${clsPrefix}-container`),
      css: getWrapperStyle({
        clsPrefix: classNamePrefix,
        theme,
        wrapperStyle,
        useNewStyles: USE_NEW_CHECKBOX_STYLES
      }),
      children: jsx(Checkbox$1, {
        checked: isChecked === null ? undefined : isChecked,
        ref: ref,
        onChange: onChange ? event => {
          onChange(event.target.checked, event);
        } : undefined,
        disabled: isDisabled,
        indeterminate: isChecked === null
        // Individual checkboxes don't depend on isHorizontal flag, orientation and spacing is handled by end users
        ,
        css: /*#__PURE__*/css(importantify(getCheckboxEmotionStyles(clsPrefix, theme, false, USE_NEW_CHECKBOX_STYLES)), process.env.NODE_ENV === "production" ? "" : ";label:DuboisCheckbox;"),
        style: style,
        "aria-checked": isChecked === null ? 'mixed' : isChecked,
        ...restProps,
        ...dangerouslySetAntdProps,
        children: jsx(RestoreAntDDefaultClsPrefix, {
          children: children
        })
      })
    })
  });
});
const CheckboxGroup = /*#__PURE__*/forwardRef(function CheckboxGroup(_ref3, ref) {
  let {
    children,
    layout = 'vertical',
    ...props
  } = _ref3;
  const {
    theme,
    getPrefixedClassName
  } = useDesignSystemTheme();
  const clsPrefix = getPrefixedClassName('checkbox');
  const {
    USE_NEW_CHECKBOX_STYLES
  } = useDesignSystemFlags();
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Checkbox$1.Group, {
      ref: ref,
      ...props,
      css: getCheckboxEmotionStyles(clsPrefix, theme, layout === 'horizontal', USE_NEW_CHECKBOX_STYLES),
      children: jsx(RestoreAntDDefaultClsPrefix, {
        children: children
      })
    })
  });
});
const CheckboxNamespace = /* #__PURE__ */Object.assign(DuboisCheckbox, {
  Group: CheckboxGroup
});
const Checkbox = CheckboxNamespace;

// TODO: I'm doing this to support storybook's docgen;
// We should remove this once we have a better storybook integration,
// since these will be exposed in the library's exports.
const __INTERNAL_DO_NOT_USE__Group = CheckboxGroup;

function getEmotionStyles(clsPrefix, theme) {
  const classFocused = `.${clsPrefix}-focused`;
  const classActiveBar = `.${clsPrefix}-active-bar`;
  const classSeparator = `.${clsPrefix}-separator`;
  const classSuffix = `.${clsPrefix}-suffix`;
  const styles = {
    height: 32,
    borderRadius: theme.borders.borderRadiusMd,
    borderColor: theme.colors.border,
    color: theme.colors.textPrimary,
    transition: 'border 0s, box-shadow 0s',
    [`&${classFocused},:hover`]: {
      borderColor: theme.colors.actionDefaultBorderHover
    },
    '&:active': {
      borderColor: theme.colors.actionDefaultBorderPress
    },
    [`&${classFocused}`]: {
      boxShadow: `none !important`,
      outline: `${theme.colors.actionDefaultBorderFocus} solid 2px !important`,
      borderColor: 'transparent !important'
    },
    [`& ${classActiveBar}`]: {
      background: `${theme.colors.actionDefaultBorderPress} !important`
    },
    [`& input::placeholder, & ${classSeparator}, & ${classSuffix}`]: {
      color: theme.colors.textPrimary
    }
  };
  return /*#__PURE__*/css(styles, process.env.NODE_ENV === "production" ? "" : ";label:getEmotionStyles;");
}
const getDropdownStyles$1 = theme => {
  return {
    zIndex: theme.options.zIndexBase + 50
  };
};
function useDatePickerStyles() {
  const {
    theme,
    getPrefixedClassName
  } = useDesignSystemTheme();
  const clsPrefix = getPrefixedClassName('picker');
  return getEmotionStyles(clsPrefix, theme);
}
const DuboisDatePicker = /*#__PURE__*/forwardRef((props, ref) => {
  const styles = useDatePickerStyles();
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(DatePicker, {
      css: styles,
      ref: ref,
      ...props,
      popupStyle: {
        ...getDropdownStyles$1(theme),
        ...(props.popupStyle || {})
      }
    })
  });
});
const RangePicker = /*#__PURE__*/forwardRef((props, ref) => {
  const styles = useDatePickerStyles();
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(DatePicker.RangePicker, {
      css: styles,
      ...props,
      ref: ref,
      popupStyle: {
        ...getDropdownStyles$1(theme),
        ...(props.popupStyle || {})
      }
    })
  });
});
const TimePicker = /*#__PURE__*/forwardRef((props, ref) => {
  const styles = useDatePickerStyles();
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(DatePicker.TimePicker, {
      css: styles,
      ...props,
      ref: ref,
      popupStyle: {
        ...getDropdownStyles$1(theme),
        ...(props.popupStyle || {})
      }
    })
  });
});
const QuarterPicker = /*#__PURE__*/forwardRef((props, ref) => {
  const styles = useDatePickerStyles();
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(DatePicker.QuarterPicker, {
      css: styles,
      ...props,
      ref: ref,
      popupStyle: {
        ...getDropdownStyles$1(theme),
        ...(props.popupStyle || {})
      }
    })
  });
});
const WeekPicker = /*#__PURE__*/forwardRef((props, ref) => {
  const styles = useDatePickerStyles();
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(DatePicker.WeekPicker, {
      css: styles,
      ...props,
      ref: ref,
      popupStyle: {
        ...getDropdownStyles$1(theme),
        ...(props.popupStyle || {})
      }
    })
  });
});
const MonthPicker = /*#__PURE__*/forwardRef((props, ref) => {
  const styles = useDatePickerStyles();
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(DatePicker.MonthPicker, {
      css: styles,
      ...props,
      ref: ref,
      popupStyle: {
        ...getDropdownStyles$1(theme),
        ...(props.popupStyle || {})
      }
    })
  });
});
const YearPicker = /*#__PURE__*/forwardRef((props, ref) => {
  const styles = useDatePickerStyles();
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(DatePicker.YearPicker, {
      css: styles,
      ...props,
      ref: ref,
      popupStyle: {
        ...getDropdownStyles$1(theme),
        ...(props.popupStyle || {})
      }
    })
  });
});

/**
 * `LegacyDatePicker` was added as a temporary solution pending an
 * official Du Bois replacement. Use with caution.
 * @deprecated
 */
const LegacyDatePicker = /* #__PURE__ */Object.assign(DuboisDatePicker, {
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
  YearPicker
});

const dialogComboboxContextDefaults = {
  label: '',
  value: [],
  isInsideDialogCombobox: false,
  multiSelect: false,
  setValue: () => {},
  setIsControlled: () => {},
  stayOpenOnSelection: false,
  setIsOpen: () => {}
};
const DialogComboboxContext = /*#__PURE__*/createContext(dialogComboboxContextDefaults);
const DialogComboboxContextProvider = _ref => {
  let {
    children,
    value
  } = _ref;
  return jsx(DialogComboboxContext.Provider, {
    value: value,
    children: children
  });
};

const useDialogComboboxContext = () => {
  return useContext(DialogComboboxContext);
};

const DialogCombobox = _ref => {
  let {
    children,
    label,
    value = [],
    open,
    ...props
  } = _ref;
  // Used to avoid infinite loop when value is controlled from within the component (DialogComboboxOptionControlledList)
  // We can't remove setValue altogether because uncontrolled component users need to be able to set the value from root for trigger to update
  const [isControlled, setIsControlled] = useState(false);
  const [selectedValue, setSelectedValue] = useState(value);
  const [isOpen, setIsOpen] = useState(Boolean(open));
  useEffect(() => {
    if ((!Array.isArray(selectedValue) || !Array.isArray(value)) && selectedValue !== value || selectedValue && value && selectedValue.length === value.length && selectedValue.every((v, i) => v === value[i])) {
      return;
    }
    if (!isControlled) {
      setSelectedValue(value);
    }
  }, [value, isControlled, selectedValue]);
  return jsx(DialogComboboxContextProvider, {
    value: {
      label,
      value: selectedValue,
      setValue: setSelectedValue,
      setIsControlled,
      isInsideDialogCombobox: true,
      multiSelect: props.multiSelect,
      stayOpenOnSelection: props.stayOpenOnSelection,
      setIsOpen
    },
    children: jsx(Root$4, {
      open: open !== undefined ? open : isOpen,
      ...props,
      children: children
    })
  });
};
const Root$4 = props => {
  const {
    children,
    stayOpenOnSelection,
    multiSelect,
    ...restProps
  } = props;
  const {
    value,
    setIsOpen
  } = useDialogComboboxContext();
  const handleOpenChange = open => {
    setIsOpen(open);
  };
  useEffect(() => {
    if (!stayOpenOnSelection && (typeof stayOpenOnSelection === 'boolean' || !multiSelect)) {
      setIsOpen(false);
    }
  }, [value, stayOpenOnSelection, multiSelect, setIsOpen]);
  return jsx(Popover$1.Root, {
    onOpenChange: handleOpenChange,
    ...restProps,
    children: children
  });
};

const getButtonContainerStyles = theme => {
  return /*#__PURE__*/css({
    display: 'flex',
    flexDirection: 'row',
    justifyContent: 'flex-end',
    alignItems: 'flex-end',
    padding: `${theme.spacing.sm}px ${theme.spacing.lg / 2}px ${theme.spacing.sm}px ${theme.spacing.lg / 2}px`,
    gap: theme.spacing.sm,
    alignSelf: 'stretch'
  }, process.env.NODE_ENV === "production" ? "" : ";label:getButtonContainerStyles;");
};
const DialogComboboxButtonContainer = _ref => {
  let {
    children,
    ...restProps
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  const {
    isInsideDialogCombobox
  } = useDialogComboboxContext();
  if (!isInsideDialogCombobox) {
    throw new Error('`DialogComboboxButtonContainer` must be used within `DialogCombobox`');
  }
  return jsx("div", {
    ...restProps,
    css: getButtonContainerStyles(theme),
    children: children
  });
};

const rotate = keyframes({
  '0%': {
    transform: 'rotate(0deg) translate3d(0, 0, 0)'
  },
  '100%': {
    transform: 'rotate(360deg) translate3d(0, 0, 0)'
  }
});
const cssSpinner = function (theme) {
  let frameRate = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : 60;
  let delay = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : 0;
  const styles = {
    animation: `${rotate} 1s steps(${frameRate}, end) infinite`,
    color: theme.colors.textSecondary,
    animationDelay: `${delay}s`,
    '@media only percy': {
      animation: 'none'
    }
  };
  return /*#__PURE__*/css(importantify(styles), process.env.NODE_ENV === "production" ? "" : ";label:cssSpinner;");
};
const Spinner = _ref => {
  let {
    frameRate,
    size = 'default',
    delay,
    className: propClass,
    ...props
  } = _ref;
  const {
    classNamePrefix,
    theme
  } = useDesignSystemTheme();

  // We use Antd classes to keep styling unchanged
  // TODO(FEINF-1407): We want to move away from Antd classes and use Emotion for styling in the future
  const sizeSuffix = size === 'small' ? '-sm' : size === 'large' ? '-lg' : '';
  const sizeClass = sizeSuffix ? `${classNamePrefix}-spin${sizeSuffix}` : '';
  const wrapperClass = `${propClass || ''} ${classNamePrefix}-spin ${sizeClass} ${classNamePrefix}-spin-spinning ${DU_BOIS_ENABLE_ANIMATION_CLASSNAME}`.trim();
  const className = `${classNamePrefix}-spin-dot ${DU_BOIS_ENABLE_ANIMATION_CLASSNAME}`.trim();
  return (
    // className has to follow {...props}, otherwise is `css` prop is passed down it will overwrite our className
    jsx("div", {
      ...props,
      className: wrapperClass,
      children: jsx(LoadingIcon, {
        css: cssSpinner(theme, frameRate, delay),
        className: className
      })
    })
  );
};

const DialogComboboxLoadingSpinner = props => {
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(Spinner, {
    css: /*#__PURE__*/css({
      display: 'flex',
      alignSelf: 'center',
      justifyContent: 'center',
      alignItems: 'center',
      height: theme.general.heightSm,
      width: theme.general.heightSm,
      '> span': {
        fontSize: 20
      }
    }, process.env.NODE_ENV === "production" ? "" : ";label:DialogComboboxLoadingSpinner;"),
    ...props
  });
};

function _EMOTION_STRINGIFIED_CSS_ERROR__$g() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const getContentWrapperStyles = (theme, _ref) => {
  let {
    maxHeight = '100vh',
    maxWidth = '100vw',
    minHeight = 0,
    minWidth = 0
  } = _ref;
  return /*#__PURE__*/css({
    maxHeight,
    maxWidth,
    minHeight,
    minWidth,
    background: theme.colors.backgroundPrimary,
    color: theme.colors.textPrimary,
    overflow: 'auto',
    // Making sure the content popover overlaps the remove button when opens to the right
    zIndex: theme.options.zIndexBase + 10,
    boxSizing: 'border-box',
    padding: `${theme.spacing.xs}px 0`,
    border: `1px solid ${theme.colors.border}`,
    boxShadow: theme.general.shadowLow,
    borderRadius: 4,
    colorScheme: theme.isDarkMode ? 'dark' : 'light'
  }, process.env.NODE_ENV === "production" ? "" : ";label:getContentWrapperStyles;");
};
var _ref3$4 = process.env.NODE_ENV === "production" ? {
  name: "1ij1o5n",
  styles: "display:flex;flex-direction:column;align-items:flex-start;justify-content:center"
} : {
  name: "189loa6-DialogComboboxContent",
  styles: "display:flex;flex-direction:column;align-items:flex-start;justify-content:center;label:DialogComboboxContent;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$g
};
const DialogComboboxContent = /*#__PURE__*/forwardRef((_ref2, forwardedRef) => {
  let {
    children,
    loading,
    maxHeight,
    maxWidth,
    minHeight,
    minWidth = 240,
    align = 'start',
    side = 'bottom',
    sideOffset = 4,
    ...restProps
  } = _ref2;
  const {
    theme
  } = useDesignSystemTheme();
  const {
    label,
    isInsideDialogCombobox
  } = useDialogComboboxContext();
  const {
    getPopupContainer
  } = useDesignSystemContext();
  if (!isInsideDialogCombobox) {
    throw new Error('`DialogComboboxContent` must be used within `DialogCombobox`');
  }
  return jsx(Popover$1.Portal, {
    container: getPopupContainer && getPopupContainer(),
    children: jsx(Popover$1.Content, {
      "aria-label": `${label} options`,
      "aria-busy": loading,
      css: getContentWrapperStyles(theme, {
        maxHeight,
        maxWidth,
        minHeight,
        minWidth
      }),
      align: align,
      side: side,
      sideOffset: sideOffset,
      ...restProps,
      ref: forwardedRef,
      children: jsx("div", {
        css: _ref3$4,
        children: loading ? jsx(DialogComboboxLoadingSpinner, {
          "aria-label": "Loading",
          alt: "Loading spinner"
        }) : children
      })
    })
  });
});

const getCountBadgeStyles = theme => /*#__PURE__*/css(importantify({
  display: 'inline-flex',
  alignItems: 'center',
  justifyContent: 'center',
  boxSizing: 'border-box',
  padding: `${theme.spacing.xs / 2}px ${theme.spacing.xs}px`,
  background: theme.colors.tagDefault,
  borderRadius: theme.general.borderRadiusBase,
  fontSize: theme.typography.fontSizeBase,
  height: 20
}), process.env.NODE_ENV === "production" ? "" : ";label:getCountBadgeStyles;");
const DialogComboboxCountBadge = props => {
  const {
    countStartAt,
    ...restOfProps
  } = props;
  const {
    theme
  } = useDesignSystemTheme();
  const {
    value
  } = useDialogComboboxContext();
  return jsx("div", {
    ...restOfProps,
    css: getCountBadgeStyles(theme),
    children: Array.isArray(value) ? countStartAt ? `+${value.length - countStartAt}` : value.length : value ? 1 : 0
  });
};

const Tooltip = _ref => {
  let {
    children,
    title,
    placement = 'top',
    dataTestId,
    dangerouslySetAntdProps,
    ...props
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  if (!title) {
    return jsx(React__default.Fragment, {
      children: children
    });
  }
  const {
    overlayInnerStyle,
    overlayStyle,
    ...delegatedDangerouslySetAntdProps
  } = dangerouslySetAntdProps || {};
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Tooltip$1
    // eslint-disable-next-line react/forbid-dom-props -- FEINF-1337 - this should turn into data-testid
    , {
      title: jsx("span", {
        "data-test-id": dataTestId,
        children: title
      }),
      placement: placement
      // Always trigger on hover and focus
      ,
      trigger: ['hover', 'focus'],
      overlayInnerStyle: {
        backgroundColor: '#2F3941',
        lineHeight: '22px',
        padding: '4px 8px',
        boxShadow: theme.general.shadowLow,
        ...overlayInnerStyle
      },
      overlayStyle: {
        zIndex: theme.options.zIndexBase + 70,
        ...overlayStyle
      },
      css: /*#__PURE__*/css({
        ...getAnimationCss(theme.options.enableAnimation)
      }, process.env.NODE_ENV === "production" ? "" : ";label:Tooltip;"),
      ...delegatedDangerouslySetAntdProps,
      ...props,
      children: children
    })
  });
};

const DialogComboboxOptionListContext = /*#__PURE__*/createContext({});
const DialogComboboxOptionListContextProvider = _ref => {
  let {
    children,
    value
  } = _ref;
  return jsx(DialogComboboxOptionListContext.Provider, {
    value: value,
    children: children
  });
};

const useDialogComboboxOptionListContext = () => {
  return useContext(DialogComboboxOptionListContext);
};

const getDialogComboboxOptionItemWrapperStyles = theme => {
  return /*#__PURE__*/css(importantify({
    display: 'flex',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'flex-start',
    alignSelf: 'stretch',
    padding: '6px 32px 6px 12px',
    lineHeight: theme.typography.lineHeightBase,
    height: theme.typography.lineHeightBase,
    boxSizing: 'content-box',
    cursor: 'pointer',
    userSelect: 'none',
    '&:hover': {
      background: theme.colors.actionTertiaryBackgroundHover
    },
    '&[disabled]': {
      pointerEvents: 'none',
      color: theme.colors.actionDisabledText,
      background: theme.colors.backgroundPrimary
    }
  }), process.env.NODE_ENV === "production" ? "" : ";label:getDialogComboboxOptionItemWrapperStyles;");
};
const infoIconStyles$1 = theme => ({
  paddingLeft: theme.spacing.xs,
  color: theme.colors.textSecondary,
  pointerEvents: 'all',
  cursor: 'pointer',
  verticalAlign: 'middle'
});

const DialogComboboxOptionListCheckboxItem = /*#__PURE__*/forwardRef((_ref, ref) => {
  let {
    value,
    checked,
    indeterminate,
    onChange,
    children,
    disabledReason,
    ...props
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  const {
    isInsideDialogComboboxOptionList
  } = useDialogComboboxOptionListContext();
  if (!isInsideDialogComboboxOptionList) {
    throw new Error('`DialogComboboxOptionListCheckboxItem` must be used within `DialogComboboxOptionList`');
  }
  const handleSelect = () => {
    if (onChange) {
      onChange(value);
    }
  };
  let content = children !== null && children !== void 0 ? children : value;
  if (props.disabled && disabledReason) {
    content = jsxs(Fragment, {
      children: [content, jsx(Tooltip, {
        title: disabledReason,
        placement: "right",
        children: jsx("span", {
          css: infoIconStyles$1(theme),
          children: jsx(InfoIcon, {})
        })
      })]
    });
  }
  return jsx("div", {
    ref: ref,
    role: "option"
    // Using aria-selected instead of aria-checked because the parent listbox
    ,
    "aria-selected": indeterminate ? false : checked,
    css: [getDialogComboboxOptionItemWrapperStyles(theme), process.env.NODE_ENV === "production" ? "" : ";label:DialogComboboxOptionListCheckboxItem;"],
    ...props,
    onClick: e => {
      if (props.disabled) {
        e.preventDefault();
      } else {
        handleSelect();
      }
    },
    children: jsx(Checkbox, {
      disabled: props.disabled,
      isChecked: indeterminate ? null : checked,
      css: /*#__PURE__*/css({
        pointerEvents: 'none',
        '& > label': {
          fontSize: theme.typography.fontSizeBase,
          fontStyle: 'normal',
          fontWeight: 400
        }
      }, process.env.NODE_ENV === "production" ? "" : ";label:DialogComboboxOptionListCheckboxItem;")
      // Needed because Antd handles keyboard inputs as clicks
      ,
      onClick: e => {
        e.stopPropagation();
        handleSelect();
      },
      children: content
    })
  });
});

const getInputEmotionStyles = (clsPrefix, theme, _ref, useNewIcons, useTransparent) => {
  let {
    validationState
  } = _ref;
  const inputClass = `.${clsPrefix}-input`;
  const affixClass = `.${clsPrefix}-input-affix-wrapper`;
  const affixClassFocused = `.${clsPrefix}-input-affix-wrapper-focused`;
  const clearIcon = `.${clsPrefix}-input-clear-icon`;
  const prefixIcon = `.${clsPrefix}-input-prefix`;
  const suffixIcon = `.${clsPrefix}-input-suffix`;
  const validationColor = getValidationStateColor(theme, validationState);
  const styles = {
    '&&': {
      lineHeight: theme.typography.lineHeightBase,
      minHeight: theme.general.heightSm,
      ...(validationState && {
        borderColor: validationColor
      }),
      '&:hover': {
        borderColor: validationState ? validationColor : theme.colors.actionPrimaryBackgroundHover
      },
      '&:focus': {
        outlineColor: validationState ? validationColor : theme.colors.actionPrimaryBackgroundDefault,
        outlineWidth: 2,
        outlineOffset: -2,
        outlineStyle: 'solid',
        boxShadow: 'none',
        borderColor: 'transparent'
      },
      '&:disabled': {
        backgroundColor: theme.colors.actionDisabledBackground,
        color: theme.colors.actionDisabledText
      }
    },
    [`&${inputClass}, ${inputClass}`]: {
      ...(useTransparent && {
        backgroundColor: 'transparent'
      })
    },
    [`&${affixClass}`]: {
      ...(useTransparent && {
        backgroundColor: 'transparent'
      }),
      lineHeight: theme.typography.lineHeightBase,
      paddingTop: 5,
      paddingBottom: 5,
      minHeight: theme.general.heightSm,
      '::before': {
        lineHeight: theme.typography.lineHeightBase
      },
      '&:hover': {
        borderColor: theme.colors.actionPrimaryBackgroundHover
      },
      [`input.${clsPrefix}-input`]: {
        borderRadius: 0
      }
    },
    [`&${affixClassFocused}`]: {
      boxShadow: 'none',
      '&&, &:focus': {
        outlineColor: theme.colors.actionPrimaryBackgroundDefault,
        outlineWidth: 2,
        outlineOffset: -2,
        outlineStyle: 'solid',
        boxShadow: 'none',
        borderColor: 'transparent'
      }
    },
    ...(useNewIcons && {
      [clearIcon]: {
        fontSize: theme.general.iconFontSizeNew
      },
      [prefixIcon]: {
        marginRight: theme.spacing.sm,
        color: theme.colors.textSecondary
      },
      [suffixIcon]: {
        marginLeft: theme.spacing.sm,
        color: theme.colors.textSecondary
      }
    }),
    ...getAnimationCss(theme.options.enableAnimation)
  };
  return /*#__PURE__*/css(importantify(styles), process.env.NODE_ENV === "production" ? "" : ";label:getInputEmotionStyles;");
};
const getInputGroupStyling = clsPrefix => {
  const inputClass = `.${clsPrefix}-input`;
  const buttonClass = `.${clsPrefix}-btn`;
  return /*#__PURE__*/css({
    display: 'inline-flex !important',
    width: 'auto',
    [`& > ${inputClass}`]: {
      flexGrow: 1,
      '&:disabled': {
        border: 'none'
      }
    },
    [`& > ${buttonClass} > span`]: {
      verticalAlign: 'middle'
    }
  }, process.env.NODE_ENV === "production" ? "" : ";label:getInputGroupStyling;");
};
const TextArea = /*#__PURE__*/forwardRef(function TextArea(_ref2, ref) {
  let {
    validationState,
    autoComplete = 'off',
    dangerouslySetAntdProps,
    dangerouslyAppendEmotionCSS,
    ...props
  } = _ref2;
  const {
    classNamePrefix,
    theme
  } = useDesignSystemTheme();
  const {
    USE_NEW_ICONS: useNewIcons,
    USE_TRANSPARENT_INPUT: useTransparent
  } = useDesignSystemFlags();
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Input$1.TextArea, {
      ref: ref,
      autoComplete: autoComplete,
      css: [getInputEmotionStyles(classNamePrefix, theme, {
        validationState
      }, useNewIcons, useTransparent), dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:TextArea;"],
      ...props,
      ...dangerouslySetAntdProps
    })
  });
});
const Password = /*#__PURE__*/forwardRef(function Password(_ref3, ref) {
  let {
    validationState,
    autoComplete = 'off',
    dangerouslySetAntdProps,
    dangerouslyAppendEmotionCSS,
    ...props
  } = _ref3;
  const {
    classNamePrefix,
    theme
  } = useDesignSystemTheme();
  const {
    USE_NEW_ICONS: useNewIcons,
    USE_TRANSPARENT_INPUT: useTransparent
  } = useDesignSystemFlags();
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Input$1.Password, {
      visibilityToggle: false,
      ref: ref,
      autoComplete: autoComplete,
      css: [getInputEmotionStyles(classNamePrefix, theme, {
        validationState
      }, useNewIcons, useTransparent), dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:Password;"],
      ...props,
      ...dangerouslySetAntdProps
    })
  });
});
const DuboisInput = /*#__PURE__*/forwardRef(function Input(_ref4, ref) {
  let {
    validationState,
    autoComplete = 'off',
    dangerouslySetAntdProps,
    dangerouslyAppendEmotionCSS,
    ...props
  } = _ref4;
  const {
    classNamePrefix,
    theme
  } = useDesignSystemTheme();
  const {
    USE_NEW_ICONS: useNewIcons,
    USE_TRANSPARENT_INPUT: useTransparent
  } = useDesignSystemFlags();
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Input$1, {
      autoComplete: autoComplete,
      ref: ref,
      css: [getInputEmotionStyles(classNamePrefix, theme, {
        validationState
      }, useNewIcons, useTransparent), dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:DuboisInput;"],
      ...props,
      ...dangerouslySetAntdProps
    })
  });
});
const Group$2 = _ref5 => {
  let {
    dangerouslySetAntdProps,
    dangerouslyAppendEmotionCSS,
    compact = true,
    ...props
  } = _ref5;
  const {
    classNamePrefix
  } = useDesignSystemTheme();
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Input$1.Group, {
      css: [getInputGroupStyling(classNamePrefix), dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:Group;"],
      compact: compact,
      ...props,
      ...dangerouslySetAntdProps
    })
  });
};

// Properly creates the namespace and dot-notation components with correct types.
const InputNamespace = /* #__PURE__ */Object.assign(DuboisInput, {
  TextArea,
  Password,
  Group: Group$2
});
const Input = InputNamespace;

// TODO: I'm doing this to support storybook's docgen;
// We should remove this once we have a better storybook integration,
// since these will be exposed in the library's exports.
const __INTERNAL_DO_NOT_USE__TextArea = TextArea;
const __INTERNAL_DO_NOT_USE__Password = Password;
const __INTERNAL_DO_NOT_USE_DEDUPE__Group = Group$2;

function _EMOTION_STRINGIFIED_CSS_ERROR__$f() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
var _ref2$5 = process.env.NODE_ENV === "production" ? {
  name: "1cjitc6",
  styles: "width:16px"
} : {
  name: "gvtsec-DialogComboboxOptionListSelectItem",
  styles: "width:16px;label:DialogComboboxOptionListSelectItem;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$f
};
const DialogComboboxOptionListSelectItem = /*#__PURE__*/forwardRef((_ref, ref) => {
  let {
    value,
    checked,
    disabledReason,
    onChange,
    children,
    ...props
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  const {
    stayOpenOnSelection,
    setIsOpen,
    value: existingValue
  } = useDialogComboboxContext();
  const {
    isInsideDialogComboboxOptionList
  } = useDialogComboboxOptionListContext();
  if (!isInsideDialogComboboxOptionList) {
    throw new Error('`DialogComboboxOptionListSelectItem` must be used within `DialogComboboxOptionList`');
  }
  const handleSelect = () => {
    if (onChange) {
      onChange(value);

      // On selecting a previously selected value, manually close the popup, top level logic will not be triggered
      if (!stayOpenOnSelection && existingValue !== null && existingValue !== void 0 && existingValue.includes(value)) {
        setIsOpen(false);
      }
    }
  };
  let content = children !== null && children !== void 0 ? children : value;
  if (props.disabled && disabledReason) {
    content = jsxs(Fragment, {
      children: [content, jsx(Tooltip, {
        title: disabledReason,
        placement: "right",
        children: jsx("span", {
          css: infoIconStyles$1(theme),
          children: jsx(InfoIcon, {})
        })
      })]
    });
  }
  return jsxs("div", {
    ref: ref,
    css: [getDialogComboboxOptionItemWrapperStyles(theme), {
      '&:focus': {
        background: theme.colors.actionTertiaryBackgroundHover,
        outline: 'none'
      }
    }, process.env.NODE_ENV === "production" ? "" : ";label:DialogComboboxOptionListSelectItem;"],
    ...props,
    onClick: e => {
      if (props.disabled) {
        e.preventDefault();
      } else {
        handleSelect();
      }
    },
    tabIndex: 0,
    onKeyDown: e => {
      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault();
          const nextSibling = e.currentTarget.nextElementSibling;
          nextSibling === null || nextSibling === void 0 ? void 0 : nextSibling.focus();
          break;
        case 'ArrowUp':
          e.preventDefault();
          const previousSibling = e.currentTarget.previousElementSibling;
          previousSibling === null || previousSibling === void 0 ? void 0 : previousSibling.focus();
          break;
        case ' ':
        case 'Enter':
          e.preventDefault();
          handleSelect();
          break;
      }
    },
    role: "option",
    "aria-selected": checked,
    "aria-label": value,
    children: [checked ? jsx(CheckIcon, {}) : jsx("div", {
      css: _ref2$5
    }), jsx("label", {
      style: {
        fontSize: theme.typography.fontSizeBase,
        fontStyle: 'normal',
        fontWeight: 400,
        marginLeft: theme.spacing.sm,
        cursor: 'pointer'
      },
      children: content
    })]
  });
});

const filterChildren = (children, searchValue) => {
  var _React$Children$map;
  return (_React$Children$map = React__default.Children.map(children, child => {
    if (child.type === DialogComboboxOptionListSelectItem || child.type === DialogComboboxOptionListCheckboxItem) {
      var _child$props, _child$props$value;
      if (child !== null && child !== void 0 && (_child$props = child.props) !== null && _child$props !== void 0 && (_child$props$value = _child$props.value) !== null && _child$props$value !== void 0 && _child$props$value.toLowerCase().includes(searchValue)) {
        return child;
      }
      return null;
    }
    return child;
  })) === null || _React$Children$map === void 0 ? void 0 : _React$Children$map.filter(child => child);
};
const DialogComboboxOptionListSearch = /*#__PURE__*/forwardRef((_ref, forwardedRef) => {
  var _filteredChildren, _filteredChildren$pro, _filteredChildren2;
  let {
    onChange,
    onSearch,
    virtualized,
    children,
    hasWrapper,
    ...restProps
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  const {
    isInsideDialogComboboxOptionList
  } = useDialogComboboxOptionListContext();
  const [searchValue, setSearchValue] = React__default.useState();
  if (!isInsideDialogComboboxOptionList) {
    throw new Error('`DialogComboboxOptionListSearch` must be used within `DialogComboboxOptionList`');
  }
  const handleOnChange = event => {
    if (!virtualized) {
      setSearchValue(event.target.value.toLowerCase());
    }
    onSearch === null || onSearch === void 0 ? void 0 : onSearch(event.target.value);
  };
  let filteredChildren = children;
  if (searchValue && !virtualized) {
    filteredChildren = filterChildren(hasWrapper ? children.props.children : children, searchValue);
    if (hasWrapper) {
      filteredChildren = /*#__PURE__*/React__default.cloneElement(children, {}, filteredChildren);
    }
  }
  return jsxs(Fragment, {
    children: [jsx("div", {
      css: /*#__PURE__*/css({
        padding: `${theme.spacing.sm}px ${theme.spacing.lg / 2}px ${theme.spacing.sm}px`,
        width: '100%',
        boxSizing: 'border-box'
      }, process.env.NODE_ENV === "production" ? "" : ";label:DialogComboboxOptionListSearch;"),
      children: jsx(Input, {
        type: "search",
        name: "search",
        ref: forwardedRef,
        prefix: jsx(SearchIcon, {}),
        placeholder: "Search",
        onChange: handleOnChange,
        value: searchValue,
        ...restProps
      })
    }), virtualized ? children : hasWrapper && (_filteredChildren = filteredChildren) !== null && _filteredChildren !== void 0 && (_filteredChildren$pro = _filteredChildren.props.children) !== null && _filteredChildren$pro !== void 0 && _filteredChildren$pro.length || !hasWrapper && (_filteredChildren2 = filteredChildren) !== null && _filteredChildren2 !== void 0 && _filteredChildren2.length ? jsx("div", {
      "aria-live": "polite",
      children: filteredChildren
    }) : jsx("div", {
      "aria-live": "assertive",
      css: /*#__PURE__*/css({
        color: theme.colors.textSecondary,
        textAlign: 'center',
        padding: '6px 32px 6px 12px',
        width: '100%',
        boxSizing: 'border-box'
      }, process.env.NODE_ENV === "production" ? "" : ";label:DialogComboboxOptionListSearch;"),
      children: "No results found"
    })]
  });
});

function _EMOTION_STRINGIFIED_CSS_ERROR__$e() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
var _ref2$4 = process.env.NODE_ENV === "production" ? {
  name: "6kz1wu",
  styles: "display:flex;flex-direction:column;align-items:flex-start"
} : {
  name: "1rbb1tc-DialogComboboxOptionControlledList",
  styles: "display:flex;flex-direction:column;align-items:flex-start;label:DialogComboboxOptionControlledList;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$e
};
const DialogComboboxOptionControlledList = /*#__PURE__*/forwardRef((_ref, forwardedRef) => {
  let {
    options,
    onChange,
    loading,
    withProgressiveLoading,
    withSearch,
    showSelectAndClearAll,
    ...restProps
  } = _ref;
  const {
    isInsideDialogCombobox,
    multiSelect,
    value,
    setValue,
    setIsControlled
  } = useDialogComboboxContext();
  if (!isInsideDialogCombobox) {
    throw new Error('`DialogComboboxOptionControlledList` must be used within `DialogCombobox`');
  }
  const isOptionChecked = options.reduce((acc, option) => {
    acc[option] = value === null || value === void 0 ? void 0 : value.includes(option);
    return acc;
  }, {});
  const handleUpdate = updatedValue => {
    setIsControlled(true);
    let newValue = [];
    if (multiSelect) {
      if (value.find(item => item === updatedValue)) {
        newValue = value.filter(item => item !== updatedValue);
      } else {
        newValue = [...value, updatedValue];
      }
    } else {
      newValue = [updatedValue];
    }
    setValue(newValue);
    isOptionChecked[updatedValue] = !isOptionChecked[updatedValue];
    if (onChange) {
      onChange(newValue);
    }
  };
  const handleSelectAll = () => {
    setIsControlled(true);
    if (value.length === options.length) {
      setValue([]);
      options.forEach(option => {
        isOptionChecked[option] = false;
      });
      if (onChange) {
        onChange([]);
      }
    } else {
      setValue(options);
      options.forEach(option => {
        isOptionChecked[option] = true;
      });
      if (onChange) {
        onChange(options);
      }
    }
  };
  const renderedOptions = jsxs(Fragment, {
    children: [showSelectAndClearAll && multiSelect && jsx(DialogComboboxOptionListCheckboxItem, {
      value: "all",
      onChange: handleSelectAll,
      checked: value.length === options.length,
      indeterminate: Boolean(value.length) && value.length !== options.length,
      children: value.length === options.length ? 'Clear all' : 'Select all'
    }), options === null || options === void 0 ? void 0 : options.map((option, key) => multiSelect ? jsx(DialogComboboxOptionListCheckboxItem, {
      value: option,
      checked: isOptionChecked[option],
      onChange: handleUpdate,
      children: option
    }, key) : jsx(DialogComboboxOptionListSelectItem, {
      value: option,
      checked: isOptionChecked[option],
      onChange: handleUpdate,
      children: option
    }, key))]
  });
  return jsx("div", {
    ref: forwardedRef,
    "aria-busy": loading,
    css: _ref2$4,
    ...restProps,
    children: jsx(DialogComboboxOptionListContextProvider, {
      value: {
        isInsideDialogComboboxOptionList: true
      },
      children: jsx(Fragment, {
        children: loading ? withProgressiveLoading ? jsxs(Fragment, {
          children: [withSearch ? jsx(DialogComboboxOptionListSearch, {
            hasWrapper: true,
            children: renderedOptions
          }) : renderedOptions, jsx(DialogComboboxLoadingSpinner, {
            "aria-label": "Loading",
            alt: "Loading spinner"
          })]
        }) : jsx(DialogComboboxLoadingSpinner, {
          "aria-label": "Loading",
          alt: "Loading spinner"
        }) : withSearch ? jsx(DialogComboboxOptionListSearch, {
          hasWrapper: true,
          children: renderedOptions
        }) : renderedOptions
      })
    })
  });
});

function _EMOTION_STRINGIFIED_CSS_ERROR__$d() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
var _ref2$3 = process.env.NODE_ENV === "production" ? {
  name: "1pgv7dg",
  styles: "display:flex;flex-direction:column;align-items:flex-start;width:100%"
} : {
  name: "1dtf9pj-DialogComboboxOptionList",
  styles: "display:flex;flex-direction:column;align-items:flex-start;width:100%;label:DialogComboboxOptionList;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$d
};
const DialogComboboxOptionList = /*#__PURE__*/forwardRef((_ref, forwardedRef) => {
  let {
    children,
    loading,
    withProgressiveLoading,
    ...restProps
  } = _ref;
  const {
    isInsideDialogCombobox
  } = useDialogComboboxContext();
  if (!isInsideDialogCombobox) {
    throw new Error('`DialogComboboxOptionList` must be used within `DialogCombobox`');
  }
  return jsx("div", {
    ref: forwardedRef,
    "aria-busy": loading,
    role: "list",
    css: _ref2$3,
    ...restProps,
    children: jsx(DialogComboboxOptionListContextProvider, {
      value: {
        isInsideDialogComboboxOptionList: true
      },
      children: loading ? withProgressiveLoading ? jsxs(Fragment, {
        children: [children, jsx(DialogComboboxLoadingSpinner, {
          "aria-label": "Loading",
          alt: "Loading spinner"
        })]
      }) : jsx(DialogComboboxLoadingSpinner, {
        "aria-label": "Loading",
        alt: "Loading spinner"
      }) : children
    })
  });
});

const DialogComboboxSectionHeader = _ref => {
  let {
    children,
    ...props
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  const {
    isInsideDialogCombobox
  } = useDialogComboboxContext();
  if (!isInsideDialogCombobox) {
    throw new Error('`DialogComboboxSectionHeader` must be used within `DialogCombobox`');
  }
  return jsx("div", {
    ...props,
    css: /*#__PURE__*/css({
      display: 'flex',
      flexDirection: 'row',
      alignItems: 'flex-start',
      padding: `${theme.spacing.xs}px ${theme.spacing.lg / 2}px`,
      alignSelf: 'stretch',
      fontWeight: 400,
      color: theme.colors.textSecondary
    }, process.env.NODE_ENV === "production" ? "" : ";label:DialogComboboxSectionHeader;"),
    children: children
  });
};

const DialogComboboxSeparator = props => {
  const {
    theme
  } = useDesignSystemTheme();
  const {
    isInsideDialogCombobox
  } = useDialogComboboxContext();
  if (!isInsideDialogCombobox) {
    throw new Error('`DialogComboboxSeparator` must be used within `DialogCombobox`');
  }
  return jsx("div", {
    ...props,
    css: /*#__PURE__*/css({
      display: 'flex',
      flexDirection: 'row',
      alignItems: 'center',
      margin: `${theme.spacing.xs}px ${theme.spacing.lg / 2}px`,
      border: `1px solid ${theme.colors.borderDecorative}`,
      borderBottom: 0,
      alignSelf: 'stretch'
    }, process.env.NODE_ENV === "production" ? "" : ";label:DialogComboboxSeparator;")
  });
};

const getTriggerWrapperStyles = (removable, width) => /*#__PURE__*/css(importantify({
  display: 'inline-flex',
  alignItems: 'center',
  ...(width && {
    width: width
  }),
  ...(removable && {
    '& > button:last-of-type': importantify({
      borderBottomLeftRadius: 0,
      borderTopLeftRadius: 0,
      marginLeft: -1
    })
  })
}), process.env.NODE_ENV === "production" ? "" : ";label:getTriggerWrapperStyles;");
const getTriggerStyles = (theme, maxWidth, removable, width) => {
  const removeButtonInteractionStyles = {
    ...(removable && {
      zIndex: theme.options.zIndexBase + 2,
      '&& + button': {
        marginLeft: -1,
        zIndex: theme.options.zIndexBase + 1
      }
    })
  };
  return /*#__PURE__*/css(importantify({
    position: 'relative',
    display: 'inline-flex',
    alignItems: 'center',
    maxWidth: maxWidth,
    justifyContent: 'flex-start',
    background: 'transparent',
    padding: '6px 8px 6px 12px',
    boxSizing: 'border-box',
    height: theme.general.heightSm,
    border: `1px solid ${theme.colors.actionDefaultBorderDefault}`,
    borderRadius: 4,
    color: theme.colors.textPrimary,
    lineHeight: theme.typography.lineHeightBase,
    cursor: 'pointer',
    ...(width && {
      width: width,
      // Only set flex: 1 to items with width, otherwise in flex containers the trigger will take up all the space and break current usages that depend on content for width
      flex: 1
    }),
    ...(removable && {
      borderBottomRightRadius: 0,
      borderTopRightRadius: 0,
      borderRightColor: 'transparent'
    }),
    '&:hover': {
      background: theme.colors.actionDefaultBackgroundHover,
      borderColor: theme.colors.actionDefaultBorderHover,
      ...removeButtonInteractionStyles
    },
    '&:focus': {
      borderColor: theme.colors.actionDefaultBorderFocus,
      ...removeButtonInteractionStyles
    },
    [`&[disabled]`]: {
      background: theme.colors.actionDisabledBackground,
      color: theme.colors.actionDisabledText,
      pointerEvents: 'none',
      userSelect: 'none'
    }
  }), process.env.NODE_ENV === "production" ? "" : ";label:getTriggerStyles;");
};
const DialogComboboxTrigger = /*#__PURE__*/forwardRef((_ref, forwardedRef) => {
  let {
    removable = false,
    onRemove,
    children,
    maxWidth = 9999,
    showTagAfterValueCount = 3,
    allowClear = true,
    controlled,
    onClear,
    wrapperProps,
    width,
    withChevronIcon = true,
    ...restProps
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  const {
    label,
    value,
    isInsideDialogCombobox,
    multiSelect,
    setValue
  } = useDialogComboboxContext();
  if (!isInsideDialogCombobox) {
    throw new Error('`DialogComboboxTrigger` must be used within `DialogCombobox`');
  }
  const handleRemove = () => {
    if (!onRemove) {
      console.warn('DialogCombobox.Trigger: Attempted remove without providing onRemove handler');
    } else {
      onRemove();
    }
  };
  const handleClear = e => {
    e.stopPropagation();
    if (controlled) {
      setValue([]);
      onClear === null || onClear === void 0 ? void 0 : onClear();
    } else if (!onClear) {
      console.warn('DialogCombobox.Trigger: Attempted clear without providing onClear handler');
    } else {
      onClear();
    }
  };
  const [showTooltip, setShowTooltip] = React__default.useState();
  const triggerContentRef = React__default.useRef(null);
  useEffect(() => {
    if ((value === null || value === void 0 ? void 0 : value.length) > showTagAfterValueCount) {
      setShowTooltip(true);
    } else if (triggerContentRef.current) {
      const {
        clientWidth,
        scrollWidth
      } = triggerContentRef.current;
      setShowTooltip(clientWidth < scrollWidth);
    }
  }, [showTagAfterValueCount, value]);
  const numValues = value.length;
  const concatenatedValues = Array.isArray(value) ? numValues > 10 ? `${value.slice(0, 10).join(', ')} + ${numValues - 10}` : value.join(', ') : value;
  const displayedValues = jsx("span", {
    children: concatenatedValues
  });
  const valuesBeforeBadge = Array.isArray(value) ? value.slice(0, showTagAfterValueCount).join(', ') : value;
  let ariaLabel = /*#__PURE__*/React__default.isValidElement(label) ? 'Dialog Combobox' : `${label}`;
  if (value !== null && value !== void 0 && value.length) {
    ariaLabel += multiSelect ? `, multiselectable, ${value.length} options selected: ${concatenatedValues}` : `, selected option: ${concatenatedValues}`;
  } else {
    ariaLabel += multiSelect ? ', multiselectable, 0 options selected' : ', no option selected';
  }
  const triggerContent = children !== null && children !== void 0 ? children : jsxs(Popover$1.Trigger, {
    "aria-label": ariaLabel,
    ref: forwardedRef,
    ...restProps,
    css: getTriggerStyles(theme, maxWidth, removable, width),
    role: "listbox",
    "aria-multiselectable": multiSelect,
    children: [jsxs("span", {
      css: /*#__PURE__*/css({
        display: 'flex',
        alignItems: 'center',
        height: theme.typography.lineHeightBase,
        marginRight: 'auto',
        '&, & > *': {
          whiteSpace: 'nowrap',
          overflow: 'hidden',
          textOverflow: 'ellipsis'
        }
      }, process.env.NODE_ENV === "production" ? "" : ";label:triggerContent;"),
      ref: triggerContentRef,
      children: [jsx("span", {
        css: /*#__PURE__*/css({
          height: theme.typography.lineHeightBase,
          marginRight: theme.spacing.xs,
          fontWeight: theme.typography.typographyBoldFontWeight,
          whiteSpace: 'unset',
          overflow: 'unset',
          textOverflow: 'unset'
        }, process.env.NODE_ENV === "production" ? "" : ";label:triggerContent;"),
        children: label
      }), (value === null || value === void 0 ? void 0 : value.length) > showTagAfterValueCount ? jsxs(Fragment, {
        children: [jsx("span", {
          style: {
            marginRight: theme.spacing.xs
          },
          children: valuesBeforeBadge
        }), jsx(DialogComboboxCountBadge, {
          countStartAt: showTagAfterValueCount,
          role: "status",
          "aria-label": "Selected options count"
        })]
      }) : displayedValues]
    }), allowClear && value !== null && value !== void 0 && value.length ? jsx(XCircleFillIcon, {
      onClick: handleClear,
      css: /*#__PURE__*/css({
        color: theme.colors.textPlaceholder,
        fontSize: theme.typography.fontSizeSm,
        marginLeft: theme.spacing.xs,
        ':hover': {
          color: theme.colors.actionTertiaryTextHover
        }
      }, process.env.NODE_ENV === "production" ? "" : ";label:triggerContent;"),
      role: "button",
      "aria-label": "Clear selection"
    }) : null, withChevronIcon ? jsx(ChevronDownIcon, {
      css: /*#__PURE__*/css({
        color: theme.colors.textSecondary,
        justifySelf: 'flex-end',
        marginLeft: theme.spacing.xs
      }, process.env.NODE_ENV === "production" ? "" : ";label:triggerContent;")
    }) : null]
  });
  return jsxs("div", {
    ...wrapperProps,
    css: [getTriggerWrapperStyles(removable, width), wrapperProps === null || wrapperProps === void 0 ? void 0 : wrapperProps.css, process.env.NODE_ENV === "production" ? "" : ";label:DialogComboboxTrigger;"],
    children: [showTooltip && value !== null && value !== void 0 && value.length ? jsx(Tooltip, {
      title: displayedValues,
      children: triggerContent
    }) : triggerContent, removable && jsx(Button, {
      "aria-label": `Remove ${label}`,
      onClick: handleRemove,
      dangerouslySetForceIconStyles: true,
      children: jsx(CloseIcon, {})
    })]
  });
});

const Spacer = _ref => {
  let {
    size = 'md',
    shrinks,
    ...props
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  const spacingValues = {
    xs: theme.spacing.xs,
    sm: theme.spacing.sm,
    md: theme.spacing.md,
    lg: theme.spacing.lg
  };
  return jsx("div", {
    css: /*#__PURE__*/css({
      height: spacingValues[size],
      ...(shrinks === false ? {
        flexShrink: 0
      } : undefined)
    }, process.env.NODE_ENV === "production" ? "" : ";label:Spacer;"),
    ...props
  });
};

const getLinkStyles = (theme, clsPrefix, useNewIcons) => {
  const classTypography = `.${clsPrefix}-typography`;
  const styles = {
    [`&${classTypography}, &${classTypography}:focus`]: {
      color: theme.colors.actionTertiaryTextDefault
    },
    [`&${classTypography}:hover, &${classTypography}:hover .anticon`]: {
      color: theme.colors.actionTertiaryTextHover,
      textDecoration: 'underline'
    },
    [`&${classTypography}:active, &${classTypography}:active .anticon`]: {
      color: theme.colors.actionTertiaryTextPress,
      textDecoration: 'underline'
    },
    [`&${classTypography}:focus-visible`]: {
      textDecoration: 'underline'
    },
    ...(useNewIcons && {
      '.anticon': {
        fontSize: 12
      }
    })
  };
  return /*#__PURE__*/css(styles, process.env.NODE_ENV === "production" ? "" : ";label:getLinkStyles;");
};
const getEllipsisNewTabLinkStyles = () => {
  const styles = {
    paddingRight: 'calc(2px + 1em)',
    // 1em for icon
    position: 'relative'
  };
  return /*#__PURE__*/css(styles, process.env.NODE_ENV === "production" ? "" : ";label:getEllipsisNewTabLinkStyles;");
};
const getIconStyles = theme => {
  const styles = {
    marginLeft: 2,
    color: theme.colors.actionTertiaryTextDefault
  };
  return /*#__PURE__*/css(styles, process.env.NODE_ENV === "production" ? "" : ";label:getIconStyles;");
};
const getEllipsisIconStyles = useNewIcons => {
  const styles = {
    position: 'absolute',
    right: 0,
    bottom: 0,
    top: 0,
    display: 'flex',
    alignItems: 'center',
    ...(useNewIcons && {
      fontSize: 12
    })
  };
  return /*#__PURE__*/css(styles, process.env.NODE_ENV === "production" ? "" : ";label:getEllipsisIconStyles;");
};
function Link(_ref) {
  let {
    dangerouslySetAntdProps,
    ...props
  } = _ref;
  const {
    children,
    openInNewTab,
    ...restProps
  } = props;
  const {
    theme,
    classNamePrefix
  } = useDesignSystemTheme();
  const {
    USE_NEW_ICONS: useNewIcons
  } = useDesignSystemFlags();
  const newTabProps = {
    rel: 'noopener noreferrer',
    target: '_blank'
  };
  const linkProps = openInNewTab ? {
    ...restProps,
    ...newTabProps
  } : {
    ...restProps
  };
  const linkStyles = props.ellipsis && openInNewTab ? [getLinkStyles(theme, classNamePrefix, useNewIcons), getEllipsisNewTabLinkStyles()] : getLinkStyles(theme, classNamePrefix, useNewIcons);
  const iconStyles = props.ellipsis ? [getIconStyles(theme), getEllipsisIconStyles()] : getIconStyles(theme);
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsxs(Typography$1.Link, {
      "aria-disabled": linkProps.disabled,
      css: linkStyles,
      ...linkProps,
      ...dangerouslySetAntdProps,
      children: [children, openInNewTab ? jsx(NewWindowIcon, {
        css: iconStyles,
        ...newTabProps
      }) : null]
    })
  });
}

const {
  Paragraph: AntDParagraph
} = Typography$1;
function getParagraphEmotionStyles(theme, props) {
  return /*#__PURE__*/css({
    '&&': {
      fontSize: theme.typography.fontSizeBase,
      lineHeight: theme.typography.lineHeightBase,
      color: getTypographyColor(theme, props.color, theme.colors.textPrimary)
    }
  }, props.disabled && {
    '&&': {
      color: theme.colors.actionDisabledText
    }
  }, props.withoutMargins && {
    '&&': {
      marginTop: 0,
      marginBottom: 0
    }
  }, process.env.NODE_ENV === "production" ? "" : ";label:getParagraphEmotionStyles;");
}
function Paragraph$1(userProps) {
  const {
    dangerouslySetAntdProps,
    withoutMargins,
    color,
    ...props
  } = userProps;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(AntDParagraph, {
      ...props,
      className: props.className,
      css: getParagraphEmotionStyles(theme, userProps),
      ...dangerouslySetAntdProps
    })
  });
}

const {
  Text: AntDText
} = Typography$1;
function getTextEmotionStyles(theme, props) {
  return /*#__PURE__*/css({
    '&&': {
      fontSize: theme.typography.fontSizeBase,
      lineHeight: theme.typography.lineHeightBase,
      color: getTypographyColor(theme, props.color, theme.colors.textPrimary)
    }
  }, props.disabled && {
    '&&': {
      color: theme.colors.actionDisabledText
    }
  }, props.hint && {
    '&&': {
      fontSize: theme.typography.fontSizeSm,
      lineHeight: theme.typography.lineHeightSm
    }
  }, props.bold && {
    '&&': {
      fontSize: theme.typography.fontSizeBase,
      fontWeight: theme.typography.typographyBoldFontWeight,
      lineHeight: theme.typography.lineHeightBase
    }
  }, props.code && {
    '&& > code': {
      fontSize: theme.typography.fontSizeBase,
      lineHeight: theme.typography.lineHeightBase,
      background: theme.colors.typographyCodeBg,
      fontFamily: 'monospace',
      borderRadius: theme.borders.borderRadiusMd,
      padding: '2px 4px',
      border: 'unset',
      margin: 0
    }
  }, props.size && {
    '&&': (() => {
      switch (props.size) {
        case 'xxl':
          return {
            fontSize: theme.typography.fontSizeXxl,
            lineHeight: theme.typography.lineHeightXxl
          };
        case 'xl':
          return {
            fontSize: theme.typography.fontSizeXl,
            lineHeight: theme.typography.lineHeightXl
          };
        case 'lg':
          return {
            fontSize: theme.typography.fontSizeLg,
            lineHeight: theme.typography.lineHeightLg
          };
        case 'sm':
          return {
            fontSize: theme.typography.fontSizeSm,
            lineHeight: theme.typography.lineHeightSm
          };
        default:
          return {
            fontSize: theme.typography.fontSizeMd,
            lineHeight: theme.typography.lineHeightMd
          };
      }
    })()
  }, props.withoutMargins && {
    '&&': {
      marginTop: 0,
      marginBottom: 0
    }
  }, process.env.NODE_ENV === "production" ? "" : ";label:getTextEmotionStyles;");
}
function Text(userProps) {
  // Omit props that are not supported by `antd`
  const {
    dangerouslySetAntdProps,
    bold,
    hint,
    withoutMargins,
    color,
    ...props
  } = userProps;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(AntDText, {
      ...props,
      className: props.className,
      css: getTextEmotionStyles(theme, userProps),
      ...dangerouslySetAntdProps
    })
  });
}

const {
  Title: AntDTitle
} = Typography$1;
function getLevelStyles(theme, props) {
  switch (props.level) {
    case 1:
      return /*#__PURE__*/css({
        '&&': {
          fontSize: theme.typography.fontSizeXxl,
          lineHeight: theme.typography.lineHeightXxl,
          fontWeight: theme.typography.typographyBoldFontWeight
        }
      }, process.env.NODE_ENV === "production" ? "" : ";label:getLevelStyles;");
    case 2:
      return /*#__PURE__*/css({
        '&&': {
          fontSize: theme.typography.fontSizeXl,
          lineHeight: theme.typography.lineHeightXl,
          fontWeight: theme.typography.typographyBoldFontWeight
        }
      }, process.env.NODE_ENV === "production" ? "" : ";label:getLevelStyles;");
    case 3:
      return /*#__PURE__*/css({
        '&&': {
          fontSize: theme.typography.fontSizeLg,
          lineHeight: theme.typography.lineHeightLg,
          fontWeight: theme.typography.typographyBoldFontWeight
        }
      }, process.env.NODE_ENV === "production" ? "" : ";label:getLevelStyles;");
    case 4:
    default:
      return /*#__PURE__*/css({
        '&&': {
          fontSize: theme.typography.fontSizeMd,
          lineHeight: theme.typography.lineHeightMd,
          fontWeight: theme.typography.typographyBoldFontWeight
        }
      }, process.env.NODE_ENV === "production" ? "" : ";label:getLevelStyles;");
  }
}
function getTitleEmotionStyles(theme, props) {
  return /*#__PURE__*/css(getLevelStyles(theme, props), {
    '&&': {
      color: getTypographyColor(theme, props.color, theme.colors.textPrimary)
    }
  }, props.withoutMargins && {
    '&&': {
      marginTop: '0 !important',
      // override general styling
      marginBottom: '0 !important' // override general styling
    }
  }, process.env.NODE_ENV === "production" ? "" : ";label:getTitleEmotionStyles;");
}
function Title$2(userProps) {
  const {
    dangerouslySetAntdProps,
    withoutMargins,
    color,
    ...props
  } = userProps;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(AntDTitle, {
      ...props,
      className: props.className,
      css: getTitleEmotionStyles(theme, userProps),
      ...dangerouslySetAntdProps
    })
  });
}

const Typography = /* #__PURE__ */(() => {
  function Typography(_ref) {
    let {
      dangerouslySetAntdProps,
      ...props
    } = _ref;
    return jsx(DesignSystemAntDConfigProvider, {
      children: jsx(Typography$1, {
        ...props,
        ...dangerouslySetAntdProps
      })
    });
  }
  Typography.Text = Text;
  Typography.Title = Title$2;
  Typography.Paragraph = Paragraph$1;
  Typography.Link = Link;
  return Typography;
})();

function _EMOTION_STRINGIFIED_CSS_ERROR__$c() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const DEFAULT_WIDTH$1 = 320;
const MIN_WIDTH = 320;
const MAX_WIDTH = '90vw';
const DEFAULT_POSITION = 'right';
const ZINDEX_OVERLAY = 10;
const ZINDEX_CONTENT = ZINDEX_OVERLAY + 10;
var _ref2$2 = process.env.NODE_ENV === "production" ? {
  name: "zh83op",
  styles: "flex-grow:1;margin-bottom:0;margin-top:0;white-space:nowrap;overflow:hidden"
} : {
  name: "h5yqvj-Content",
  styles: "flex-grow:1;margin-bottom:0;margin-top:0;white-space:nowrap;overflow:hidden;label:Content;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$c
};
const Content$4 = _ref => {
  let {
    children,
    footer,
    title,
    width,
    position: positionOverride,
    useCustomScrollBehavior,
    expandContentToFullHeight,
    disableOpenAutoFocus,
    onInteractOutside
  } = _ref;
  const {
    getPopupContainer
  } = useDesignSystemContext();
  const {
    theme
  } = useDesignSystemTheme();
  const horizontalContentPadding = theme.spacing.lg;
  const contentContainerRef = useRef(null);
  const position = positionOverride !== null && positionOverride !== void 0 ? positionOverride : DEFAULT_POSITION;
  const overlayShow = position === 'right' ? keyframes({
    '0%': {
      transform: 'translate(100%, 0)'
    },
    '100%': {
      transform: 'translate(0, 0)'
    }
  }) : keyframes({
    '0%': {
      transform: 'translate(-100%, 0)'
    },
    '100%': {
      transform: 'translate(0, 0)'
    }
  });
  const dialogPrimitiveContentStyle = /*#__PURE__*/css({
    color: theme.colors.textPrimary,
    backgroundColor: theme.colors.backgroundPrimary,
    boxShadow: 'hsl(206 22% 7% / 35%) 0px 10px 38px -10px, hsl(206 22% 7% / 20%) 0px 10px 20px -15px',
    position: 'fixed',
    top: 0,
    left: position === 'left' ? 0 : undefined,
    right: position === 'right' ? 0 : undefined,
    boxSizing: 'border-box',
    width: width !== null && width !== void 0 ? width : DEFAULT_WIDTH$1,
    minWidth: MIN_WIDTH,
    maxWidth: MAX_WIDTH,
    zIndex: theme.options.zIndexBase + ZINDEX_CONTENT,
    height: '100vh',
    paddingTop: theme.spacing.md,
    paddingLeft: 0,
    paddingBottom: 0,
    paddingRight: 0,
    overflow: 'hidden',
    '&:focus': {
      outline: 'none'
    },
    '@media (prefers-reduced-motion: no-preference)': {
      animation: `${overlayShow} 350ms cubic-bezier(0.16, 1, 0.3, 1)`
    }
  }, process.env.NODE_ENV === "production" ? "" : ";label:dialogPrimitiveContentStyle;");
  return jsxs(DialogPrimitive.Portal, {
    container: getPopupContainer && getPopupContainer(),
    children: [jsx(DialogPrimitive.Overlay, {
      css: /*#__PURE__*/css({
        backgroundColor: theme.colors.overlayOverlay,
        position: 'fixed',
        inset: 0,
        // needed so that it covers the PersonaNavSidebar
        zIndex: theme.options.zIndexBase + ZINDEX_OVERLAY
      }, process.env.NODE_ENV === "production" ? "" : ";label:Content;")
    }), jsx(DialogPrimitive.DialogContent, {
      css: dialogPrimitiveContentStyle,
      style: {
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'flex-start'
      },
      ref: contentContainerRef,
      onOpenAutoFocus: event => {
        if (disableOpenAutoFocus) {
          event.preventDefault();
        }
      },
      onInteractOutside: onInteractOutside,
      children: jsxs(ApplyDesignSystemContextOverrides, {
        getPopupContainer: () => {
          var _contentContainerRef$;
          return (_contentContainerRef$ = contentContainerRef.current) !== null && _contentContainerRef$ !== void 0 ? _contentContainerRef$ : document.body;
        },
        children: [jsxs("div", {
          css: /*#__PURE__*/css({
            flexGrow: 0,
            flexShrink: 1,
            display: 'flex',
            flexDirection: 'row',
            justifyContent: 'space-between',
            alignItems: 'center',
            paddingRight: horizontalContentPadding,
            paddingLeft: horizontalContentPadding,
            marginBottom: theme.spacing.sm
          }, process.env.NODE_ENV === "production" ? "" : ";label:Content;"),
          children: [jsx(DialogPrimitive.Title, {
            title: typeof title === 'string' ? title : undefined,
            asChild: typeof title === 'string',
            css: _ref2$2,
            children: typeof title === 'string' ? jsx(Typography.Title, {
              level: 2,
              withoutMargins: true,
              ellipsis: true,
              children: title
            }) : title
          }), jsx(DialogPrimitive.Close, {
            asChild: true,
            css: /*#__PURE__*/css({
              flexShrink: 1,
              marginLeft: theme.spacing.xs
            }, process.env.NODE_ENV === "production" ? "" : ";label:Content;"),
            children: jsx(Button, {
              "aria-label": "Close",
              icon: jsx(CloseIcon, {})
            })
          })]
        }), jsxs("div", {
          css: /*#__PURE__*/css({
            // in order to have specific content in the drawer scroll with fixed title
            // hide overflow here and remove padding on the right side; content will be responsible for setting right padding
            // so that the scrollbar will appear in the padding right gutter
            paddingRight: useCustomScrollBehavior ? 0 : horizontalContentPadding,
            paddingLeft: horizontalContentPadding,
            overflowY: useCustomScrollBehavior ? 'hidden' : 'auto',
            height: expandContentToFullHeight ? '100%' : undefined,
            ...(theme.isDarkMode === false && !useCustomScrollBehavior ? {
              // Achieves an inner shadow on the content, but only when there is more left to scroll. When the content fits
              // in the container without scrolling, no shadow will be shown.
              // Taken from: https://css-tricks.com/scroll-shadows-with-javascript/
              background: `linear-gradient(
                    white 30%,
                    rgba(255, 255, 255, 0)
                  ) center top,
  
                  linear-gradient(
                    rgba(255, 255, 255, 0),
                    white 70%
                  ) center bottom,
      
                  radial-gradient(
                    farthest-side at 50% 0,
                    rgba(0, 0, 0, 0.2),
                    rgba(0, 0, 0, 0)
                  ) center top,
      
                  radial-gradient(
                    farthest-side at 50% 100%,
                    rgba(0, 0, 0, 0.2),
                    rgba(0, 0, 0, 0)
                  ) center bottom`,
              backgroundRepeat: 'no-repeat',
              backgroundSize: '100% 40px, 100% 40px, 100% 14px, 100% 14px',
              backgroundAttachment: 'local, local, scroll, scroll',
              backgroundOrigin: 'content-box'
            } : {})
          }, process.env.NODE_ENV === "production" ? "" : ";label:Content;"),
          children: [children, !footer && jsx(Spacer, {
            size: "lg"
          })]
        }), footer && jsx("div", {
          style: {
            paddingTop: theme.spacing.md,
            paddingRight: horizontalContentPadding,
            paddingLeft: horizontalContentPadding,
            paddingBottom: theme.spacing.lg,
            flexGrow: 0,
            flexShrink: 1
          },
          children: footer
        })]
      })
    })]
  });
};
function Root$3(props) {
  return jsx(DialogPrimitive.Root, {
    ...props
  });
}
function Trigger$2(props) {
  return jsx(DialogPrimitive.Trigger, {
    asChild: true,
    ...props
  });
}

var Drawer = /*#__PURE__*/Object.freeze({
  __proto__: null,
  Content: Content$4,
  Root: Root$3,
  Trigger: Trigger$2
});

/**
 * @deprecated Use `DropdownMenu` instead.
 */
const Dropdown = _ref => {
  let {
    dangerouslySetAntdProps,
    ...props
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Dropdown$1, {
      mouseLeaveDelay: 0.25,
      ...props,
      overlayStyle: {
        zIndex: theme.options.zIndexBase + 50,
        ...props.overlayStyle
      },
      ...dangerouslySetAntdProps
    })
  });
};

const Root$2 = DropdownMenu$1.Root; // Behavioral component only

const Content$3 = /*#__PURE__*/forwardRef(function Content(_ref, ref) {
  let {
    children,
    minWidth = 220,
    ...props
  } = _ref;
  const {
    getPopupContainer
  } = useDesignSystemContext();
  return jsx(DropdownMenu$1.Portal, {
    container: getPopupContainer && getPopupContainer(),
    children: jsx(DropdownMenu$1.Content, {
      ref: ref,
      loop: true,
      css: [contentStyles$1, {
        minWidth
      }, process.env.NODE_ENV === "production" ? "" : ";label:Content;"],
      sideOffset: 4,
      align: "start",
      ...props,
      children: children
    })
  });
});
const SubContent = /*#__PURE__*/forwardRef(function Content(_ref2, ref) {
  let {
    children,
    minWidth = 220,
    ...props
  } = _ref2;
  const {
    getPopupContainer
  } = useDesignSystemContext();
  return jsx(DropdownMenu$1.Portal, {
    container: getPopupContainer && getPopupContainer(),
    children: jsx(DropdownMenu$1.SubContent, {
      ref: ref,
      loop: true,
      css: [contentStyles$1, {
        minWidth
      }, process.env.NODE_ENV === "production" ? "" : ";label:SubContent;"],
      sideOffset: -2,
      alignOffset: -5,
      ...props,
      children: children
    })
  });
});
const Trigger$1 = /*#__PURE__*/forwardRef(function Trigger(_ref3, ref) {
  let {
    children,
    ...props
  } = _ref3;
  return jsx(DropdownMenu$1.Trigger, {
    ref: ref,
    ...props,
    children: children
  });
});
const Item = /*#__PURE__*/forwardRef(function Item(_ref4, ref) {
  let {
    children,
    disabledReason,
    danger,
    onClick,
    ...props
  } = _ref4;
  const itemRef = useRef(null);
  useImperativeHandle(ref, () => itemRef.current);
  return jsx(DropdownMenu$1.Item, {
    css: theme => [itemStyles, danger && dangerItemStyles(theme)],
    ref: itemRef,
    onClick: e => {
      if (props.disabled) {
        e.preventDefault();
      } else {
        onClick === null || onClick === void 0 ? void 0 : onClick(e);
      }
    },
    ...props,
    children: getNewChildren(children, props, disabledReason, itemRef)
  });
});
const Label$1 = /*#__PURE__*/forwardRef(function Label(_ref5, ref) {
  let {
    children,
    ...props
  } = _ref5;
  return jsx(DropdownMenu$1.Label, {
    ref: ref,
    css: [itemStyles, theme => ({
      color: theme.colors.textSecondary,
      '&:hover': {
        cursor: 'default'
      }
    }), process.env.NODE_ENV === "production" ? "" : ";label:Label;"],
    ...props,
    children: children
  });
});
const Separator = /*#__PURE__*/forwardRef(function Separator(_ref6, ref) {
  let {
    children,
    ...props
  } = _ref6;
  return jsx(DropdownMenu$1.Separator, {
    ref: ref,
    css: theme => ({
      height: 1,
      margin: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
      backgroundColor: theme.colors.borderDecorative
    }),
    ...props,
    children: children
  });
});
const SubTrigger = /*#__PURE__*/forwardRef(function TriggerItem(_ref7, ref) {
  let {
    children,
    disabledReason,
    ...props
  } = _ref7;
  const subTriggerRef = useRef(null);
  useImperativeHandle(ref, () => subTriggerRef.current);
  return jsxs(DropdownMenu$1.SubTrigger, {
    ref: subTriggerRef,
    css: [itemStyles, theme => ({
      '&[data-state="open"]': {
        backgroundColor: theme.colors.actionTertiaryBackgroundHover
      }
    }), process.env.NODE_ENV === "production" ? "" : ";label:SubTrigger;"],
    ...props,
    children: [getNewChildren(children, props, disabledReason, subTriggerRef), jsx(HintColumn, {
      css: theme => ({
        margin: CONSTANTS$1.subMenuIconMargin(theme),
        display: 'flex',
        alignSelf: 'stretch',
        alignItems: 'center'
      }),
      children: jsx(ChevronRightIcon, {
        css: theme => ({
          fontSize: CONSTANTS$1.subMenuIconSize(theme)
        })
      })
    })]
  });
});

/**
 * Deprecated. Use `SubTrigger` instead.
 * @deprecated
 */
const TriggerItem = SubTrigger;
const CheckboxItem = /*#__PURE__*/forwardRef(function CheckboxItem(_ref8, ref) {
  let {
    children,
    disabledReason,
    ...props
  } = _ref8;
  const flags = useDesignSystemFlags();
  const checkboxItemRef = useRef(null);
  useImperativeHandle(ref, () => checkboxItemRef.current);
  return jsx(DropdownMenu$1.CheckboxItem, {
    ref: checkboxItemRef,
    css: theme => [itemStyles, checkboxItemStyles(theme, flags)],
    ...props,
    children: getNewChildren(children, props, disabledReason, checkboxItemRef)
  });
});
const ItemIndicator = /*#__PURE__*/forwardRef(function ItemIndicator(_ref9, ref) {
  let {
    children,
    ...props
  } = _ref9;
  const flags = useDesignSystemFlags();
  return jsx(DropdownMenu$1.ItemIndicator, {
    ref: ref,
    css: theme => ({
      marginLeft: -(CONSTANTS$1.checkboxIconWidth(theme, flags) + CONSTANTS$1.checkboxPaddingRight(theme, flags)),
      position: 'absolute',
      ...(!flags.USE_NEW_ICONS && {
        fontSize: 24
      })
    }),
    ...props,
    children: children !== null && children !== void 0 ? children : jsx(CheckIcon, {
      css: theme => ({
        color: theme.colors.textSecondary
      })
    })
  });
});
const Arrow$1 = /*#__PURE__*/forwardRef(function Arrow(_ref10, ref) {
  let {
    children,
    ...props
  } = _ref10;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(DropdownMenu$1.Arrow, {
    css: /*#__PURE__*/css({
      fill: theme.colors.backgroundPrimary,
      stroke: theme.colors.borderDecorative,
      strokeDashoffset: -CONSTANTS$1.arrowBottomLength(),
      strokeDasharray: CONSTANTS$1.arrowBottomLength() + 2 * CONSTANTS$1.arrowSide(),
      strokeWidth: CONSTANTS$1.arrowStrokeWidth(),
      // TODO: This is a temporary fix for the alignment of the Arrow;
      // Radix has changed the implementation for v1.0.0 (uses floating-ui)
      // which has new behaviors for alignment that we don't want. Generally
      // we need to fix the arrow to always be aligned to the left of the menu (with
      // offset equal to border radius)
      position: 'relative',
      top: -1
    }, process.env.NODE_ENV === "production" ? "" : ";label:Arrow;"),
    ref: ref,
    width: 12,
    height: 6,
    ...props,
    children: children
  });
});
const RadioItem = /*#__PURE__*/forwardRef(function RadioItem(_ref11, ref) {
  let {
    children,
    disabledReason,
    ...props
  } = _ref11;
  const flags = useDesignSystemFlags();
  const radioItemRef = useRef(null);
  useImperativeHandle(ref, () => radioItemRef.current);
  return jsx(DropdownMenu$1.RadioItem, {
    ref: radioItemRef,
    css: theme => [itemStyles, checkboxItemStyles(theme, flags)],
    ...props,
    children: getNewChildren(children, props, disabledReason, radioItemRef)
  });
});

// UNWRAPPED RADIX-UI-COMPONENTS
const Group$1 = DropdownMenu$1.Group;
const RadioGroup = DropdownMenu$1.RadioGroup;
const Sub = DropdownMenu$1.Sub;

// EXTRA COMPONENTS
const HintColumn = /*#__PURE__*/forwardRef(function HintColumn(_ref12, ref) {
  let {
    children,
    ...props
  } = _ref12;
  return jsx("div", {
    ref: ref,
    css: [metaTextStyles, "margin-left:auto;" + (process.env.NODE_ENV === "production" ? "" : ";label:HintColumn;")],
    ...props,
    children: children
  });
});
const HintRow = /*#__PURE__*/forwardRef(function HintRow(_ref13, ref) {
  let {
    children,
    ...props
  } = _ref13;
  return jsx("div", {
    ref: ref,
    css: [metaTextStyles, "min-width:100%;" + (process.env.NODE_ENV === "production" ? "" : ";label:HintRow;")],
    ...props,
    children: children
  });
});
const IconWrapper = /*#__PURE__*/forwardRef(function IconWrapper(_ref14, ref) {
  let {
    children,
    ...props
  } = _ref14;
  return jsx("div", {
    ref: ref,
    css: theme => ({
      fontSize: 16,
      color: theme.colors.textSecondary,
      paddingRight: 4
    }),
    ...props,
    children: children
  });
});
const getNewChildren = (children, props, disabledReason, ref) => {
  const childCount = Children.count(children);
  const tooltip = jsx(Tooltip, {
    title: disabledReason,
    placement: "right",
    dangerouslySetAntdProps: {
      getPopupContainer: () => ref.current || document.body
    },
    children: jsx("span", {
      css: theme => infoIconStyles(theme),
      children: jsx(InfoIcon, {})
    })
  });
  if (childCount === 1) {
    return getChild(children, Boolean(props['disabled']), disabledReason, tooltip, 0, childCount);
  }
  return Children.map(children, (child, idx) => {
    return getChild(child, Boolean(props['disabled']), disabledReason, tooltip, idx, childCount);
  });
};
const getChild = (child, isDisabled, disabledReason, tooltip, index, siblingCount) => {
  const HintColumnType = jsx(HintColumn, {}).type;
  const isHintColumnType = Boolean(child && typeof child !== 'string' && typeof child !== 'number' && typeof child !== 'boolean' && 'type' in child && (child === null || child === void 0 ? void 0 : child.type) === HintColumnType);
  if (isDisabled && disabledReason && child && isHintColumnType) {
    return jsxs(Fragment, {
      children: [tooltip, child]
    });
  } else if (index === siblingCount - 1 && isDisabled && disabledReason) {
    return jsxs(Fragment, {
      children: [child, tooltip]
    });
  }
  return child;
};

// CONSTANTS
const CONSTANTS$1 = {
  itemPaddingVertical(theme) {
    // The number from the mocks is the midpoint between constants
    return 0.5 * theme.spacing.xs + 0.5 * theme.spacing.sm;
  },
  itemPaddingHorizontal(theme) {
    return theme.spacing.sm;
  },
  checkboxIconWidth(theme, flags) {
    return flags.USE_NEW_ICONS ? theme.general.iconFontSizeNew : theme.spacing.lg;
  },
  checkboxPaddingLeft(theme) {
    return theme.spacing.sm;
  },
  checkboxPaddingRight(theme, flags) {
    return flags.USE_NEW_ICONS ? theme.spacing.sm : theme.spacing.xs;
  },
  subMenuIconMargin(theme) {
    // Negative margin so the icons can be larger without increasing the overall item height
    const iconMarginVertical = this.itemPaddingVertical(theme) / 2;
    const iconMarginRight = -this.itemPaddingVertical(theme) + theme.spacing.sm * 1.5;
    return `${-iconMarginVertical}px ${-iconMarginRight}px ${-iconMarginVertical}px auto`;
  },
  subMenuIconSize(theme) {
    return theme.spacing.lg;
  },
  arrowBottomLength() {
    // The built in arrow is a polygon: 0,0 30,0 15,10
    return 30;
  },
  arrowHeight() {
    return 10;
  },
  arrowSide() {
    return 2 * (this.arrowHeight() ** 2 * 2) ** 0.5;
  },
  arrowStrokeWidth() {
    // This is eyeballed b/c relative to the svg viewbox coordinate system
    return 2;
  }
};
const dropdownContentStyles = theme => ({
  backgroundColor: theme.colors.backgroundPrimary,
  color: theme.colors.textPrimary,
  lineHeight: theme.typography.lineHeightBase,
  border: `1px solid ${theme.colors.borderDecorative}`,
  borderRadius: theme.borders.borderRadiusMd,
  padding: `${theme.spacing.xs}px 0`,
  boxShadow: theme.general.shadowLow,
  userSelect: 'none',
  // Ant Design uses 1000s for their zIndex space; this ensures Radix works with that, but
  // we'll likely need to be sure that all Radix components are using the same zIndex going forward.
  //
  // Additionally, there is an issue where macOS overlay scrollbars in Chrome and Safari (sometimes!)
  // overlap other elements with higher zIndex, because the scrollbars themselves have zIndex 9999,
  // so we have to use a higher value than that: https://github.com/databricks/universe/pull/232825
  zIndex: 10000,
  a: importantify({
    color: theme.colors.textPrimary,
    '&:hover, &:focus': {
      color: theme.colors.textPrimary
    }
  })
});
const contentStyles$1 = theme => ({
  ...dropdownContentStyles(theme)
});
const itemStyles = theme => ({
  padding: `${CONSTANTS$1.itemPaddingVertical(theme)}px ${CONSTANTS$1.itemPaddingHorizontal(theme)}px`,
  display: 'flex',
  flexWrap: 'wrap',
  alignItems: 'center',
  outline: 'unset',
  '&:hover': {
    cursor: 'pointer'
  },
  '&:focus': {
    backgroundColor: theme.colors.actionTertiaryBackgroundHover
  },
  '&[data-disabled]': {
    pointerEvents: 'none',
    color: theme.colors.actionDisabledText
  }
});
const dangerItemStyles = theme => ({
  color: theme.colors.textValidationDanger,
  '&:hover, &:focus': {
    backgroundColor: theme.colors.actionDangerDefaultBackgroundHover
  }
});
const infoIconStyles = theme => ({
  display: 'inline-flex',
  paddingLeft: theme.spacing.xs,
  color: theme.colors.textSecondary,
  pointerEvents: 'all'
});
const checkboxItemStyles = (theme, flags) => ({
  paddingLeft: CONSTANTS$1.checkboxIconWidth(theme, flags) + CONSTANTS$1.checkboxPaddingLeft(theme) + CONSTANTS$1.checkboxPaddingRight(theme, flags)
});
const metaTextStyles = theme => ({
  color: theme.colors.textSecondary,
  fontSize: theme.typography.fontSizeSm,
  '[data-disabled] &': {
    color: theme.colors.actionDisabledText
  }
});

var DropdownMenu = /*#__PURE__*/Object.freeze({
  __proto__: null,
  Root: Root$2,
  Content: Content$3,
  SubContent: SubContent,
  Trigger: Trigger$1,
  Item: Item,
  Label: Label$1,
  Separator: Separator,
  SubTrigger: SubTrigger,
  TriggerItem: TriggerItem,
  CheckboxItem: CheckboxItem,
  ItemIndicator: ItemIndicator,
  Arrow: Arrow$1,
  RadioItem: RadioItem,
  Group: Group$1,
  RadioGroup: RadioGroup,
  Sub: Sub,
  HintColumn: HintColumn,
  HintRow: HintRow,
  IconWrapper: IconWrapper,
  dropdownContentStyles: dropdownContentStyles
});

function _EMOTION_STRINGIFIED_CSS_ERROR__$b() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const {
  Title: Title$1,
  Paragraph
} = Typography;
function getEmptyStyles(theme) {
  const styles = {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    textAlign: 'center',
    maxWidth: 600,
    wordBreak: 'break-word',
    // Set size of image to 64px
    '> :first-child': {
      fontSize: 64,
      color: theme.colors.actionDisabledText,
      marginBottom: theme.spacing.md
    }
  };
  return /*#__PURE__*/css(styles, process.env.NODE_ENV === "production" ? "" : ";label:getEmptyStyles;");
}
function getEmptyTitleStyles(theme, clsPrefix) {
  const styles = {
    [`&.${clsPrefix}-typography`]: {
      color: theme.colors.textSecondary,
      marginTop: 0,
      marginBottom: 0
    }
  };
  return /*#__PURE__*/css(styles, process.env.NODE_ENV === "production" ? "" : ";label:getEmptyTitleStyles;");
}
function getEmptyDescriptionStyles(theme, clsPrefix) {
  const styles = {
    [`&.${clsPrefix}-typography`]: {
      color: theme.colors.textSecondary,
      marginBottom: theme.spacing.md
    }
  };
  return /*#__PURE__*/css(styles, process.env.NODE_ENV === "production" ? "" : ";label:getEmptyDescriptionStyles;");
}
var _ref$2 = process.env.NODE_ENV === "production" ? {
  name: "zl1inp",
  styles: "display:flex;justify-content:center"
} : {
  name: "11tid6c-Empty",
  styles: "display:flex;justify-content:center;label:Empty;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$b
};
const Empty = props => {
  const {
    theme,
    classNamePrefix
  } = useDesignSystemTheme();
  const {
    title,
    description,
    image = jsx(GridDashIcon, {}),
    button,
    dangerouslyAppendEmotionCSS,
    ...dataProps
  } = props;
  return jsx("div", {
    ...dataProps,
    css: _ref$2,
    children: jsxs("div", {
      css: [getEmptyStyles(theme), dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:Empty;"],
      children: [image, title && jsx(Title$1, {
        level: 3,
        css: getEmptyTitleStyles(theme, classNamePrefix),
        children: title
      }), jsx(Paragraph, {
        css: getEmptyDescriptionStyles(theme, classNamePrefix),
        children: description
      }), button]
    })
  });
};

const getFormItemEmotionStyles = _ref => {
  let {
    theme,
    clsPrefix,
    useNewIcons
  } = _ref;
  const clsFormItemLabel = `.${clsPrefix}-form-item-label`;
  const clsFormItemInputControl = `.${clsPrefix}-form-item-control-input`;
  const clsFormItemExplain = `.${clsPrefix}-form-item-explain`;
  const clsHasError = `.${clsPrefix}-form-item-has-error`;
  return /*#__PURE__*/css({
    [clsFormItemLabel]: {
      fontWeight: theme.typography.typographyBoldFontWeight,
      lineHeight: theme.typography.lineHeightBase,
      ...(useNewIcons && {
        '.anticon': {
          fontSize: theme.general.iconFontSizeNew
        }
      })
    },
    [clsFormItemExplain]: {
      fontSize: theme.typography.fontSizeSm,
      margin: 0
    },
    [clsFormItemInputControl]: {
      minHeight: theme.general.heightSm
    },
    [`${clsFormItemInputControl} input[disabled]`]: {
      border: 'none'
    },
    [`&${clsHasError} input:focus`]: importantify({
      boxShadow: 'none'
    }),
    ...getAnimationCss(theme.options.enableAnimation)
  }, process.env.NODE_ENV === "production" ? "" : ";label:getFormItemEmotionStyles;");
};
const FormDubois = /*#__PURE__*/forwardRef(function Form(_ref2, ref) {
  let {
    dangerouslySetAntdProps,
    children,
    ...props
  } = _ref2;
  const mergedProps = {
    ...props,
    layout: props.layout || 'vertical',
    requiredMark: props.requiredMark || false
  };
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Form$1, {
      ...mergedProps,
      colon: false,
      ref: ref,
      ...dangerouslySetAntdProps,
      children: jsx(RestoreAntDDefaultClsPrefix, {
        children: children
      })
    })
  });
});
const FormItem = _ref3 => {
  let {
    dangerouslySetAntdProps,
    children,
    ...props
  } = _ref3;
  const {
    theme,
    classNamePrefix
  } = useDesignSystemTheme();
  const {
    USE_NEW_ICONS: useNewIcons
  } = useDesignSystemFlags();
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Form$1.Item, {
      ...props,
      css: getFormItemEmotionStyles({
        theme,
        clsPrefix: classNamePrefix,
        useNewIcons
      }),
      ...dangerouslySetAntdProps,
      children: children
    })
  });
};
const FormNamespace = /* #__PURE__ */Object.assign(FormDubois, {
  Item: FormItem,
  List: Form$1.List,
  useForm: Form$1.useForm
});
const Form = FormNamespace;

// TODO: I'm doing this to support storybook's docgen;
// We should remove this once we have a better storybook integration,
// since these will be exposed in the library's exports.
const __INTERNAL_DO_NOT_USE__FormItem = FormItem;

const getMessageStyles = (clsPrefix, theme) => {
  const errorClass = `.${clsPrefix}-form-error-message`;
  const successClass = `.${clsPrefix}-form-success-message`;
  const warningClass = `.${clsPrefix}-form-warning-message`;
  const styles = {
    '&&': {
      lineHeight: theme.typography.lineHeightSm,
      fontSize: theme.typography.fontSizeSm,
      marginTop: theme.spacing.sm,
      display: 'flex',
      alignItems: 'start'
    },
    [`&${errorClass}`]: {
      color: theme.colors.actionDangerPrimaryBackgroundDefault
    },
    [`&${successClass}`]: {
      color: theme.colors.textValidationSuccess
    },
    [`&${warningClass}`]: {
      color: theme.colors.textValidationWarning
    },
    ...getAnimationCss(theme.options.enableAnimation)
  };
  return /*#__PURE__*/css(styles, process.env.NODE_ENV === "production" ? "" : ";label:getMessageStyles;");
};
const VALIDATION_STATE_ICONS = {
  error: DangerIcon,
  success: CheckCircleIcon,
  warning: WarningIcon
};
function FormMessage(_ref) {
  let {
    message,
    type = 'error',
    className = '',
    css
  } = _ref;
  const {
    classNamePrefix,
    theme
  } = useDesignSystemTheme();
  const stateClass = `${classNamePrefix}-form-${type}-message`;
  const StateIcon = VALIDATION_STATE_ICONS[type];
  const wrapperClass = `${classNamePrefix}-form-message ${className} ${stateClass}`.trim();
  return jsxs("div", {
    className: wrapperClass,
    css: [getMessageStyles(classNamePrefix, theme), css, process.env.NODE_ENV === "production" ? "" : ";label:FormMessage;"],
    children: [jsx(StateIcon, {}), jsx("div", {
      style: {
        paddingLeft: theme.spacing.xs
      },
      children: message
    })]
  });
}

const getHintStyles = (classNamePrefix, theme) => {
  const styles = {
    display: 'block',
    color: theme.colors.textSecondary,
    lineHeight: theme.typography.lineHeightSm,
    fontSize: theme.typography.fontSizeSm,
    [`&& + .${classNamePrefix}-input, && + .${classNamePrefix}-select, && + .${classNamePrefix}-checkbox-group, && + .${classNamePrefix}-radio-group`]: {
      marginTop: theme.spacing.sm
    }
  };
  return /*#__PURE__*/css(styles, process.env.NODE_ENV === "production" ? "" : ";label:getHintStyles;");
};
const Hint = props => {
  const {
    classNamePrefix,
    theme
  } = useDesignSystemTheme();
  const {
    className,
    ...restProps
  } = props;
  return jsx("span", {
    className: classnames(`${classNamePrefix}-hint`, className),
    css: getHintStyles(classNamePrefix, theme),
    ...restProps
  });
};

const getLabelStyles = (classNamePrefix, theme, _ref) => {
  let {
    inline
  } = _ref;
  const styles = {
    '&&': {
      color: theme.colors.textPrimary,
      fontWeight: theme.typography.typographyBoldFontWeight,
      display: inline ? 'inline' : 'block',
      lineHeight: theme.typography.lineHeightBase
    },
    [`&& + .${classNamePrefix}-input, && + .${classNamePrefix}-select, && + .${classNamePrefix}-checkbox-group, && + .${classNamePrefix}-radio-group`]: {
      marginTop: theme.spacing.sm
    }
  };
  return /*#__PURE__*/css(styles, process.env.NODE_ENV === "production" ? "" : ";label:getLabelStyles;");
};
const Label = props => {
  const {
    children,
    className,
    inline,
    ...restProps
  } = props;
  const {
    classNamePrefix,
    theme
  } = useDesignSystemTheme();
  return jsx("label", {
    css: getLabelStyles(classNamePrefix, theme, {
      inline
    }),
    className: classnames(`${classNamePrefix}-label`, className),
    ...restProps,
    children: children
  });
};

const getRadioInputStyles = _ref => {
  let {
    clsPrefix,
    theme,
    useNewStyles
  } = _ref;
  return {
    [`.${clsPrefix}`]: {
      alignSelf: 'start',
      // Unchecked Styles
      [`> .${clsPrefix}-input + .${clsPrefix}-inner`]: {
        width: theme.spacing.md,
        height: theme.spacing.md,
        background: useNewStyles ? theme.colors.actionDefaultBackgroundDefault : theme.colors.radioDefaultBackground,
        borderStyle: 'solid',
        borderColor: useNewStyles ? theme.colors.actionDefaultBorderDefault : theme.colors.radioDefaultBorder,
        boxShadow: 'unset',
        transform: 'unset',
        // This prevents an awkward jitter on the border
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        borderRadius: '50%',
        '&:after': {
          all: 'unset'
        }
      },
      // Hover
      [`&:not(.${clsPrefix}-disabled) > .${clsPrefix}-input:hover + .${clsPrefix}-inner`]: {
        borderColor: useNewStyles ? theme.colors.actionPrimaryBackgroundHover : theme.colors.radioInteractiveHover,
        background: useNewStyles ? theme.colors.actionDefaultBackgroundHover : theme.colors.radioInteractiveHoverSecondary
      },
      // Focus
      [`&:not(.${clsPrefix}-disabled)> .${clsPrefix}-input:focus + .${clsPrefix}-inner`]: {
        borderColor: useNewStyles ? theme.colors.actionPrimaryBackgroundDefault : theme.colors.primary
      },
      // Active
      [`&:not(.${clsPrefix}-disabled)> .${clsPrefix}-input:active + .${clsPrefix}-inner`]: {
        borderColor: useNewStyles ? theme.colors.actionPrimaryBackgroundPress : theme.colors.radioInteractivePress,
        background: useNewStyles ? theme.colors.actionDefaultBackgroundPress : theme.colors.radioInteractivePressSecondary
      },
      // Disabled
      [`&.${clsPrefix}-disabled > .${clsPrefix}-input + .${clsPrefix}-inner`]: {
        ...(useNewStyles ? {
          border: `none !important`,
          // Ant uses !important
          background: theme.colors.actionDisabledBackground
        } : {
          borderColor: `${theme.colors.radioDisabled}!important` // Ant uses !important
        })
      },

      // Checked Styles
      [`&.${clsPrefix}-checked`]: {
        '&:after': {
          border: 'unset'
        },
        [`> .${clsPrefix}-input + .${clsPrefix}-inner`]: {
          background: useNewStyles ? theme.colors.actionPrimaryBackgroundDefault : theme.colors.primary,
          borderColor: theme.colors.primary,
          '&:after': {
            content: `""`,
            borderRadius: theme.spacing.xs,
            backgroundColor: useNewStyles ? theme.colors.white : theme.colors.radioDefaultBackground,
            width: theme.spacing.xs,
            height: theme.spacing.xs
          }
        },
        // Hover
        [`&:not(.${clsPrefix}-disabled) > .${clsPrefix}-input:hover + .${clsPrefix}-inner`]: {
          background: useNewStyles ? theme.colors.actionPrimaryBackgroundHover : theme.colors.radioInteractiveHover,
          borderColor: useNewStyles ? theme.colors.actionPrimaryBackgroundPress : theme.colors.radioInteractiveHover
        },
        // Focus
        [`&:not(.${clsPrefix}-disabled) > .${clsPrefix}-input:focus-visible + .${clsPrefix}-inner`]: {
          background: useNewStyles ? theme.colors.actionDefaultBackgroundPress : theme.colors.primary,
          borderColor: useNewStyles ? theme.colors.actionDefaultBorderFocus : theme.colors.primary,
          boxShadow: `0 0 0 1px ${useNewStyles ? theme.colors.actionDefaultBackgroundDefault : theme.colors.radioDefaultBackground}, 0 0 0 3px ${theme.colors.primary}`
        },
        // Active
        [`&:not(.${clsPrefix}-disabled) > .${clsPrefix}-input:active + .${clsPrefix}-inner`]: {
          background: useNewStyles ? theme.colors.actionDefaultBackgroundPress : theme.colors.radioInteractivePress,
          borderColor: useNewStyles ? theme.colors.actionDefaultBorderPress : theme.colors.radioInteractivePress
        },
        // Disabled
        [`&.${clsPrefix}-disabled > .${clsPrefix}-input + .${clsPrefix}-inner`]: {
          background: useNewStyles ? theme.colors.actionDisabledBackground : theme.colors.radioDisabled,
          border: useNewStyles ? 'none !important' : `2px solid ${theme.colors.radioDisabled}!important` // !important inherited from ant
        }
      }
    }
  };
};

const getCommonRadioGroupStyles = _ref2 => {
  let {
    theme,
    clsPrefix,
    classNamePrefix,
    useNewStyles
  } = _ref2;
  return /*#__PURE__*/css({
    '& > label': {
      ...(useNewStyles && {
        [`&.${classNamePrefix}-radio-wrapper-disabled > span`]: {
          color: theme.colors.textPrimary
        }
      })
    },
    [`& > label + .${classNamePrefix}-hint`]: {
      paddingLeft: theme.spacing.lg
    },
    ...getRadioInputStyles({
      theme,
      clsPrefix,
      useNewStyles
    }),
    ...getAnimationCss(theme.options.enableAnimation)
  }, process.env.NODE_ENV === "production" ? "" : ";label:getCommonRadioGroupStyles;");
};
const getHorizontalRadioGroupStyles = _ref3 => {
  let {
    theme,
    classNamePrefix,
    useNewStyles
  } = _ref3;
  return /*#__PURE__*/css({
    '&&': {
      display: 'grid',
      gridTemplateRows: '[label] auto [hint] auto',
      gridAutoColumns: 'max-content',
      gridColumnGap: useNewStyles ? theme.spacing.md : theme.spacing.sm
    },
    '& > label': {
      gridRow: 'label / label',
      marginRight: 0
    },
    [`& > label + .${classNamePrefix}-hint`]: {
      display: 'inline-block',
      gridRow: 'hint / hint'
    }
  }, process.env.NODE_ENV === "production" ? "" : ";label:getHorizontalRadioGroupStyles;");
};
const getVerticalRadioGroupStyles = _ref4 => {
  let {
    theme,
    classNamePrefix,
    useNewStyles
  } = _ref4;
  return /*#__PURE__*/css({
    display: 'flex',
    flexDirection: 'column',
    flexWrap: 'wrap',
    '& > label': {
      fontWeight: 'normal',
      ...(useNewStyles && {
        paddingBottom: theme.spacing.sm
      })
    },
    [`& > label:last-of-type`]: {
      paddingBottom: 0
    },
    [`& > label + .${classNamePrefix}-hint`]: {
      marginBottom: theme.spacing.sm,
      paddingLeft: theme.spacing.lg,
      ...(useNewStyles && {
        marginTop: `-${theme.spacing.sm}px`
      })
    },
    [`& > label:last-of-type + .${classNamePrefix}-hint`]: {
      ...(useNewStyles && {
        marginTop: 0
      })
    }
  }, process.env.NODE_ENV === "production" ? "" : ";label:getVerticalRadioGroupStyles;");
};
const getRadioStyles = _ref5 => {
  let {
    theme,
    clsPrefix,
    useNewStyles
  } = _ref5;
  // Default as bold for standalone radios
  const fontWeight = 'normal';
  const styles = {
    fontWeight
  };
  return /*#__PURE__*/css({
    ...getRadioInputStyles({
      theme,
      clsPrefix,
      useNewStyles
    }),
    ...styles
  }, process.env.NODE_ENV === "production" ? "" : ";label:getRadioStyles;");
};
const DuboisRadio = /*#__PURE__*/forwardRef(function Radio(_ref6, ref) {
  let {
    children,
    dangerouslySetAntdProps,
    ...props
  } = _ref6;
  const {
    theme,
    getPrefixedClassName
  } = useDesignSystemTheme();
  const {
    USE_NEW_RADIO_STYLES
  } = useDesignSystemFlags();
  const clsPrefix = getPrefixedClassName('radio');
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Radio$1, {
      css: getRadioStyles({
        theme,
        clsPrefix,
        useNewStyles: USE_NEW_RADIO_STYLES
      }),
      ...props,
      ...dangerouslySetAntdProps,
      ref: ref,
      children: jsx(RestoreAntDDefaultClsPrefix, {
        children: children
      })
    })
  });
});
const StyledRadioGroup = /*#__PURE__*/forwardRef(function StyledRadioGroup(_ref7, ref) {
  let {
    children,
    dangerouslySetAntdProps,
    ...props
  } = _ref7;
  const {
    theme,
    getPrefixedClassName,
    classNamePrefix
  } = useDesignSystemTheme();
  const {
    USE_NEW_RADIO_STYLES
  } = useDesignSystemFlags();
  const clsPrefix = getPrefixedClassName('radio');
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Radio$1.Group, {
      ...props,
      css: getCommonRadioGroupStyles({
        theme,
        clsPrefix,
        classNamePrefix,
        useNewStyles: USE_NEW_RADIO_STYLES
      }),
      ...dangerouslySetAntdProps,
      ref: ref,
      children: jsx(RestoreAntDDefaultClsPrefix, {
        children: children
      })
    })
  });
});
const HorizontalGroup = /*#__PURE__*/forwardRef(function HorizontalGroup(_ref8, ref) {
  let {
    dangerouslySetAntdProps,
    ...props
  } = _ref8;
  const {
    theme,
    classNamePrefix
  } = useDesignSystemTheme();
  const {
    USE_NEW_RADIO_STYLES
  } = useDesignSystemFlags();
  return jsx(StyledRadioGroup, {
    css: getHorizontalRadioGroupStyles({
      theme,
      classNamePrefix,
      useNewStyles: USE_NEW_RADIO_STYLES
    }),
    ...props,
    ref: ref,
    ...dangerouslySetAntdProps
  });
});
const Group = /*#__PURE__*/forwardRef(function HorizontalGroup(_ref9, ref) {
  let {
    dangerouslySetAntdProps,
    layout = 'vertical',
    ...props
  } = _ref9;
  const {
    theme,
    classNamePrefix
  } = useDesignSystemTheme();
  const {
    USE_NEW_RADIO_STYLES
  } = useDesignSystemFlags();
  return jsx(StyledRadioGroup, {
    css: layout === 'horizontal' ? getHorizontalRadioGroupStyles({
      theme,
      classNamePrefix,
      useNewStyles: USE_NEW_RADIO_STYLES
    }) : getVerticalRadioGroupStyles({
      theme,
      classNamePrefix,
      useNewStyles: USE_NEW_RADIO_STYLES
    }),
    ...props,
    ref: ref,
    ...dangerouslySetAntdProps
  });
});

// Note: We are overriding ant's default "Group" with our own.
const RadioNamespace = /* #__PURE__ */Object.assign(DuboisRadio, {
  Group,
  HorizontalGroup
});
const Radio = RadioNamespace;

// TODO: I'm doing this to support storybook's docgen;
// We should remove this once we have a better storybook integration,
// since these will be exposed in the library's exports.
// We should ideally be using __Group instead of __VerticalGroup, but that exists under Checkbox too and conflicts, therefore
// we show a wrong component name in "Show code" in docs, fix included in story to replace this with correct name
const __INTERNAL_DO_NOT_USE__VerticalGroup = Group;
const __INTERNAL_DO_NOT_USE__HorizontalGroup = HorizontalGroup;

function getSelectEmotionStyles(_ref) {
  let {
    clsPrefix,
    theme,
    validationState,
    useNewIcons
  } = _ref;
  const classFocused = `.${clsPrefix}-focused`;
  const classOpen = `.${clsPrefix}-open`;
  const classSingle = `.${clsPrefix}-single`;
  const classSelector = `.${clsPrefix}-selector`;
  const classDisabled = `.${clsPrefix}-disabled`;
  const classMultiple = `.${clsPrefix}-multiple`;
  const classItem = `.${clsPrefix}-selection-item`;
  const classItemOverflowContainer = `.${clsPrefix}-selection-overflow`;
  const classItemOverflowItem = `.${clsPrefix}-selection-overflow-item`;
  const classItemOverflowSuffix = `.${clsPrefix}-selection-overflow-item-suffix`;
  const classArrow = `.${clsPrefix}-arrow`;
  const classArrowLoading = `.${clsPrefix}-arrow-loading`;
  const classPlaceholder = `.${clsPrefix}-selection-placeholder`;
  const classCloseButton = `.${clsPrefix}-selection-item-remove`;
  const classSearch = `.${clsPrefix}-selection-search`;
  const classShowSearch = `.${clsPrefix}-show-search`;
  const classSearchClear = `.${clsPrefix}-clear`;
  const classAllowClear = `.${clsPrefix}-allow-clear`;
  const classSearchInput = `.${clsPrefix}-selection-search-input`;
  const validationColor = getValidationStateColor(theme, validationState);
  const styles = {
    '&:hover': {
      [classSelector]: {
        borderColor: theme.colors.actionDefaultBorderHover
      }
    },
    [classSelector]: {
      paddingLeft: 12,
      // Only the select _item_ is clickable, so we need to have zero padding here, and add it on the item itself,
      // to make sure the whole select is clickable.
      paddingRight: 0,
      color: theme.colors.textPrimary,
      backgroundColor: 'transparent',
      height: theme.general.heightSm,
      '::after': {
        lineHeight: theme.typography.lineHeightBase
      },
      '::before': {
        lineHeight: theme.typography.lineHeightBase
      }
    },
    [classSingle]: {
      [`&${classSelector}`]: {
        height: theme.general.heightSm
      }
    },
    [classItem]: {
      color: theme.colors.textPrimary,
      paddingRight: 32,
      lineHeight: theme.typography.lineHeightBase,
      paddingTop: 5,
      paddingBottom: 5
    },
    // Note: This supports search, which we don't support. The styles here support legacy usages.
    [classSearch]: {
      right: 24,
      left: 8,
      marginInlineStart: 4,
      [classSearchInput]: {
        color: theme.colors.actionDefaultTextDefault,
        height: 24
      }
    },
    [`&${classSingle}`]: {
      [classSearchInput]: {
        height: theme.general.heightSm
      }
    },
    // Note: This supports search, which we don't support. The styles here support legacy usages.
    [`&${classShowSearch}${classOpen}${classSingle}`]: {
      [classItem]: {
        color: theme.colors.actionDisabledText
      }
    },
    // Note: This supports search, which we don't support. The styles here support legacy usages.
    [classSearchClear]: {
      right: useNewIcons ? 24 : 32
    },
    [`&${classFocused}`]: {
      [classSelector]: {
        outlineColor: theme.colors.actionPrimaryBackgroundDefault,
        outlineWidth: 2,
        outlineOffset: -2,
        outlineStyle: 'solid',
        borderColor: 'transparent',
        boxShadow: 'none'
      }
    },
    [`&${classDisabled}`]: {
      [classSelector]: {
        backgroundColor: theme.colors.actionDisabledBackground,
        color: theme.colors.actionDisabledText,
        border: 'transparent'
      },
      [classItem]: {
        color: theme.colors.actionDisabledText
      },
      [classArrow]: {
        color: theme.colors.actionDisabledText
      }
    },
    [classArrow]: {
      height: useNewIcons ? theme.general.iconFontSizeNew : theme.general.iconSize,
      width: useNewIcons ? theme.general.iconFontSizeNew : theme.general.iconSize,
      top: useNewIcons ? (theme.general.heightSm - theme.general.iconFontSizeNew) / 2 : 4,
      marginTop: 0,
      color: theme.colors.textSecondary,
      ...(useNewIcons && {
        fontSize: theme.general.iconFontSizeNew
      }),
      '.anticon': {
        // For some reason ant sets this to 'auto'. Need to set it back to 'none' to allow the element below to receive
        // the click event.
        pointerEvents: 'none'
      },
      [`&${classArrowLoading}`]: {
        top: useNewIcons ? (theme.general.heightSm - theme.general.iconFontSizeNew) / 2 : 4,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: useNewIcons ? theme.general.iconFontSizeNew : theme.general.iconSize
      }
    },
    [classPlaceholder]: {
      color: theme.colors.textSecondary,
      right: 'auto',
      left: 'auto',
      width: '100%',
      paddingRight: 32,
      lineHeight: theme.typography.lineHeightBase,
      alignSelf: 'center'
    },
    [`&${classMultiple}`]: {
      [classSelector]: {
        paddingTop: 3,
        paddingBottom: 3,
        paddingLeft: 8,
        paddingRight: 30,
        minHeight: theme.general.heightSm,
        height: 'auto',
        '&::after': {
          margin: 0
        }
      },
      [classItem]: {
        backgroundColor: theme.colors.tagDefault,
        color: theme.colors.textPrimary,
        border: 'none',
        height: 20,
        lineHeight: theme.typography.lineHeightBase,
        fontSize: theme.typography.fontSizeBase,
        marginInlineEnd: 4,
        marginTop: 2,
        marginBottom: 2,
        paddingRight: 0,
        paddingTop: 0,
        paddingBottom: 0
      },
      [classItemOverflowContainer]: {
        minHeight: 24
      },
      [classItemOverflowItem]: {
        alignSelf: 'auto',
        height: 24,
        lineHeight: theme.typography.lineHeightBase
      },
      [classSearch]: {
        marginTop: 0,
        left: 0,
        right: 0
      },
      [`&${classDisabled}`]: {
        [classItem]: {
          paddingRight: 2
        }
      },
      [classArrow]: {
        top: useNewIcons ? (theme.general.heightSm - theme.general.iconFontSizeNew) / 2 : 5
      },
      [`&${classAllowClear}`]: {
        [classSearchClear]: {
          top: useNewIcons ? (theme.general.heightSm - theme.general.iconFontSizeNew + 4) / 2 : 16
        }
      },
      [classPlaceholder]: {
        // Compensate for the caret placeholder width
        paddingLeft: 4,
        color: theme.colors.textPlaceholder
      },
      [`&:not(${classFocused})`]: {
        [classItemOverflowSuffix]: {
          // Do not keep the caret's placeholder at full height when not focused,
          // because it introduces a new line even when not focused. Using display: none would break the caret
          height: 0
        }
      }
    },
    [`&${classMultiple}${classDisabled}`]: {
      [classItem]: {
        color: theme.colors.actionDisabledText
      }
    },
    [`&${classAllowClear}`]: {
      [classItem]: {
        paddingRight: 0
      },
      [classSelector]: {
        paddingRight: 52
      },
      [classSearchClear]: {
        top: useNewIcons ? (theme.general.heightSm - theme.general.iconFontSizeNew + 4) / 2 : 16,
        opacity: 100,
        ...(useNewIcons && {
          width: theme.general.iconFontSizeNew,
          height: theme.general.iconFontSizeNew,
          marginTop: 0
        })
      }
    },
    [classCloseButton]: {
      color: theme.colors.textPrimary,
      borderTopRightRadius: theme.borders.borderRadiusMd,
      borderBottomRightRadius: theme.borders.borderRadiusMd,
      height: useNewIcons ? theme.general.iconFontSizeNew : 20,
      width: useNewIcons ? theme.general.iconFontSizeNew : 20,
      lineHeight: theme.typography.lineHeightBase,
      paddingInlineEnd: 0,
      marginInlineEnd: useNewIcons ? 0 : -2,
      ...(useNewIcons && {
        '& > .anticon': {
          height: theme.general.iconFontSizeNew - 4,
          fontSize: theme.general.iconFontSizeNew - 4
        }
      }),
      '&:hover': {
        color: theme.colors.actionTertiaryTextHover,
        backgroundColor: theme.colors.tagHover
      },
      '&:active': {
        color: theme.colors.actionTertiaryTextPress,
        backgroundColor: theme.colors.tagPress
      }
    },
    ...(validationState && {
      [`& > ${classSelector}`]: {
        borderColor: validationColor,
        '&:hover': {
          borderColor: validationColor
        }
      },
      [`&${classFocused} > ${classSelector}`]: {
        outlineColor: validationColor,
        outlineOffset: -2
      }
    }),
    ...getAnimationCss(theme.options.enableAnimation)
  };
  const importantStyles = importantify(styles);
  return /*#__PURE__*/css(importantStyles, process.env.NODE_ENV === "production" ? "" : ";label:getSelectEmotionStyles;");
}
function getDropdownStyles(clsPrefix, theme) {
  let useNewDropdownStyle = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : false;
  let useNewIcons = arguments.length > 3 ? arguments[3] : undefined;
  const classItem = `.${clsPrefix}-item-option`;
  const classItemActive = `.${clsPrefix}-item-option-active`;
  const classItemSelected = `.${clsPrefix}-item-option-selected`;
  const classItemState = `.${clsPrefix}-item-option-state`;
  const classItemContent = `.${clsPrefix}-item-option-content`;
  const CONTENT_LEFT_PADDING = 28;
  const styles = {
    ...(useNewDropdownStyle ? dropdownContentStyles(theme) : {
      borderColor: theme.colors.borderDecorative,
      borderWidth: 1,
      borderStyle: 'solid',
      zIndex: theme.options.zIndexBase + 50,
      boxShadow: theme.general.shadowLow
    }),
    [classItem]: {
      height: theme.general.heightSm,
      ...(useNewDropdownStyle && {
        padding: '4px 8px',
        alignItems: 'center',
        lineHeight: theme.typography.lineHeightBase
      })
    },
    ...(useNewDropdownStyle && {
      [classItemContent]: {
        paddingLeft: CONTENT_LEFT_PADDING
      }
    }),
    [classItemActive]: {
      backgroundColor: theme.colors.actionTertiaryBackgroundHover,
      height: theme.general.heightSm,
      '&:hover': {
        backgroundColor: theme.colors.actionTertiaryBackgroundHover
      }
    },
    [classItemSelected]: {
      backgroundColor: theme.colors.actionTertiaryBackgroundHover,
      fontWeight: 'normal',
      '&:hover': {
        backgroundColor: theme.colors.actionTertiaryBackgroundHover
      }
    },
    [classItemState]: {
      color: theme.colors.textSecondary,
      ...(useNewDropdownStyle && {
        position: 'absolute'
      }),
      ...(useNewIcons && {
        '& > span': {
          verticalAlign: 'middle'
        }
      })
    },
    [`.${clsPrefix}-loading-options`]: {
      pointerEvents: 'none',
      margin: '0 auto',
      height: theme.general.heightSm,
      display: 'block',
      ...(useNewDropdownStyle && {
        left: -CONTENT_LEFT_PADDING / 2
      })
    },
    ...getAnimationCss(theme.options.enableAnimation)
  };
  const importantStyles = importantify(styles);
  return /*#__PURE__*/css(importantStyles, process.env.NODE_ENV === "production" ? "" : ";label:getDropdownStyles;");
}
function getLoadingIconStyles(theme) {
  return /*#__PURE__*/css({
    fontSize: 20,
    color: theme.colors.textSecondary,
    lineHeight: '20px'
  }, process.env.NODE_ENV === "production" ? "" : ";label:getLoadingIconStyles;");
}
const scrollbarVisibleItemsCount = 8;
const getIconSizeStyle = (theme, useNewIcons, defaultIconSize, newIconDefault) => importantify({
  fontSize: useNewIcons ? newIconDefault !== null && newIconDefault !== void 0 ? newIconDefault : theme.general.iconFontSizeNew : defaultIconSize !== null && defaultIconSize !== void 0 ? defaultIconSize : theme.general.iconFontSize
});
function DuboisSelect(_ref2, ref) {
  let {
    children,
    validationState,
    loading,
    mode,
    options,
    notFoundContent,
    optionFilterProp,
    dangerouslySetAntdProps,
    virtual,
    dropdownClassName,
    ...restProps
  } = _ref2;
  const {
    theme,
    getPrefixedClassName
  } = useDesignSystemTheme();
  const {
    USE_NEW_SELECT_DROPDOWN_STYLES,
    USE_NEW_ICONS: useNewIcons
  } = useDesignSystemFlags();
  const clsPrefix = getPrefixedClassName('select');
  return jsx(ClassNames, {
    children: _ref3 => {
      let {
        css: css$1
      } = _ref3;
      return jsx(DesignSystemAntDConfigProvider, {
        children: jsx(Select$1, {
          css: getSelectEmotionStyles({
            clsPrefix,
            theme,
            validationState,
            useNewIcons
          }),
          removeIcon: jsx(CloseIcon, {
            css: getIconSizeStyle(theme, useNewIcons, 20)
          }),
          clearIcon: useNewIcons ? jsx(XCircleFillIcon, {
            css: getIconSizeStyle(theme, useNewIcons, 16, 12),
            "aria-label": "close-circle"
          }) : undefined,
          ref: ref,
          suffixIcon: loading && mode === 'tags' ? jsx(LoadingIcon, {
            spin: true,
            "aria-label": "loading",
            css: getIconSizeStyle(theme, useNewIcons, 13, 12)
          }) : jsx(ChevronDownIcon, {
            css: getIconSizeStyle(theme, useNewIcons, 24)
          }),
          menuItemSelectedIcon: jsx(CheckIcon, {
            css: getIconSizeStyle(theme, useNewIcons, 24)
          }),
          showArrow: true,
          dropdownMatchSelectWidth: true,
          notFoundContent: notFoundContent !== null && notFoundContent !== void 0 ? notFoundContent : jsx("div", {
            css: /*#__PURE__*/css({
              color: theme.colors.textSecondary,
              textAlign: 'center'
            }),
            children: "No results found"
          }),
          dropdownClassName: css$1([getDropdownStyles(clsPrefix, theme, USE_NEW_SELECT_DROPDOWN_STYLES, useNewIcons), dropdownClassName]),
          maxTagPlaceholder: items => `+ ${items.length} more`,
          mode: mode,
          options: options,
          loading: loading,
          filterOption: true
          // NOTE(FEINF-1102): This is needed to avoid ghost scrollbar that generates error when clicked on exactly 8 elements
          // Because by default AntD uses true for virtual, we want to replicate the same even if there are no children
          ,
          virtual: virtual !== null && virtual !== void 0 ? virtual : children && Array.isArray(children) && children.length !== scrollbarVisibleItemsCount || options && options.length !== scrollbarVisibleItemsCount || !children && !options,
          optionFilterProp: optionFilterProp !== null && optionFilterProp !== void 0 ? optionFilterProp : 'children',
          ...restProps,
          ...dangerouslySetAntdProps,
          children: loading && mode !== 'tags' ? jsxs(Fragment, {
            children: [children, jsx(Option, {
              disabled: true,
              value: "select-loading-options",
              className: `${clsPrefix}-loading-options`,
              children: jsx(LoadingIcon, {
                spin: true,
                css: getLoadingIconStyles(theme),
                "aria-label": "loading"
              })
            })]
          }) : children
        })
      });
    }
  });
}
const SelectOption = /*#__PURE__*/forwardRef(function Option(props, ref) {
  const {
    dangerouslySetAntdProps,
    ...restProps
  } = props;
  return jsx(Select$1.Option, {
    ...restProps,
    ref: ref,
    ...dangerouslySetAntdProps
  });
});

// Needed for rc-select to not throw warning about our component not being Select.Option
SelectOption.isSelectOption = true;

/**
 * @deprecated use Select.Option instead
 */
const Option = SelectOption;
const SelectOptGroup = /* #__PURE__ */(() => {
  const OptGroup = /*#__PURE__*/forwardRef(function OptGroup(props, ref) {
    return jsx(Select$1.OptGroup, {
      ...props,
      isSelectOptGroup: true,
      ref: ref
    });
  });
  // Needed for antd to work properly and for rc-select to not throw warning about our component not being Select.OptGroup
  OptGroup.isSelectOptGroup = true;
  return OptGroup;
})();

/**
 * @deprecated use Select.OptGroup instead
 */
const OptGroup = SelectOptGroup;
const Select = /* #__PURE__ */(() => {
  const DuboisRefForwardedSelect = /*#__PURE__*/forwardRef(DuboisSelect);
  DuboisRefForwardedSelect.Option = SelectOption;
  DuboisRefForwardedSelect.OptGroup = SelectOptGroup;
  return DuboisRefForwardedSelect;
})();

function RHFControlledInput(_ref) {
  let {
    name,
    control,
    rules,
    ...restProps
  } = _ref;
  const {
    field
  } = useController({
    name: name,
    control: control,
    rules: rules
  });
  return jsx(Input, {
    ...restProps,
    ...field,
    value: field.value,
    defaultValue: restProps.defaultValue
  });
}
function RHFControlledPasswordInput(_ref2) {
  let {
    name,
    control,
    rules,
    ...restProps
  } = _ref2;
  const {
    field
  } = useController({
    name: name,
    control: control,
    rules: rules
  });
  return jsx(Input.Password, {
    ...restProps,
    ...field,
    value: field.value,
    defaultValue: restProps.defaultValue
  });
}
function RHFControlledTextArea(_ref3) {
  let {
    name,
    control,
    rules,
    ...restProps
  } = _ref3;
  const {
    field
  } = useController({
    name: name,
    control: control,
    rules: rules
  });
  return jsx(Input.TextArea, {
    ...restProps,
    ...field,
    value: field.value,
    defaultValue: restProps.defaultValue
  });
}
function RHFControlledSelect(_ref4) {
  let {
    name,
    control,
    rules,
    ...restProps
  } = _ref4;
  const {
    field
  } = useController({
    name: name,
    control: control,
    rules: rules
  });
  return jsx(Select, {
    ...restProps,
    ...field,
    value: field.value,
    defaultValue: field.value
  });
}
function RHFControlledCheckboxGroup(_ref5) {
  let {
    name,
    control,
    rules,
    ...restProps
  } = _ref5;
  const {
    field
  } = useController({
    name: name,
    control: control,
    rules: rules
  });
  return jsx(Checkbox.Group, {
    ...restProps,
    ...field,
    value: field.value
  });
}
function RHFControlledCheckbox(_ref6) {
  let {
    name,
    control,
    rules,
    ...restProps
  } = _ref6;
  const {
    field
  } = useController({
    name: name,
    control: control,
    rules: rules
  });
  const {
    theme
  } = useDesignSystemTheme();
  return jsx("div", {
    css: /*#__PURE__*/css({
      marginTop: theme.spacing.sm
    }, process.env.NODE_ENV === "production" ? "" : ";label:RHFControlledCheckbox;"),
    children: jsx(Checkbox, {
      ...restProps,
      ...field,
      checked: field.value
    })
  });
}
function RHFControlledRadioGroup(_ref7) {
  let {
    name,
    control,
    rules,
    ...restProps
  } = _ref7;
  const {
    field
  } = useController({
    name: name,
    control: control,
    rules: rules
  });
  return jsx(Radio.Group, {
    ...restProps,
    ...field
  });
}
const RHFControlledComponents = {
  Input: RHFControlledInput,
  Password: RHFControlledPasswordInput,
  TextArea: RHFControlledTextArea,
  Select: RHFControlledSelect,
  Checkbox: RHFControlledCheckbox,
  CheckboxGroup: RHFControlledCheckboxGroup,
  RadioGroup: RHFControlledRadioGroup
};

const FormUI = {
  Message: FormMessage,
  Label: Label,
  Hint: Hint
};

const Col = _ref => {
  let {
    dangerouslySetAntdProps,
    ...props
  } = _ref;
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Col$1, {
      ...props,
      ...dangerouslySetAntdProps
    })
  });
};

const ROW_GUTTER_SIZE = 8;
const Row = _ref => {
  let {
    gutter = ROW_GUTTER_SIZE,
    ...props
  } = _ref;
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Row$1, {
      gutter: gutter,
      ...props
    })
  });
};

const Space = _ref => {
  let {
    dangerouslySetAntdProps,
    ...props
  } = _ref;
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Space$1, {
      ...props,
      ...dangerouslySetAntdProps
    })
  });
};

function _EMOTION_STRINGIFIED_CSS_ERROR__$a() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const getHeaderStyles = (clsPrefix, theme) => {
  const breadcrumbClass = `.${clsPrefix}-breadcrumb`;
  const styles = {
    [breadcrumbClass]: {
      lineHeight: theme.typography.lineHeightBase
    }
  };
  return /*#__PURE__*/css(importantify(styles), process.env.NODE_ENV === "production" ? "" : ";label:getHeaderStyles;");
};
var _ref$1 = process.env.NODE_ENV === "production" ? {
  name: "5s4ezj",
  styles: "display:inline-flex;vertical-align:middle;align-items:center;margin-left:8px"
} : {
  name: "gxikdi-titleAddOnsWrapper",
  styles: "display:inline-flex;vertical-align:middle;align-items:center;margin-left:8px;label:titleAddOnsWrapper;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$a
};
var _ref2$1 = process.env.NODE_ENV === "production" ? {
  name: "1q4vxyr",
  styles: "margin-left:8px"
} : {
  name: "ozrfom-buttonContainer",
  styles: "margin-left:8px;label:buttonContainer;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$a
};
var _ref3$3 = process.env.NODE_ENV === "production" ? {
  name: "s079uh",
  styles: "margin-top:2px"
} : {
  name: "1ky5whb-titleIfOtherElementsPresent",
  styles: "margin-top:2px;label:titleIfOtherElementsPresent;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$a
};
var _ref4$1 = process.env.NODE_ENV === "production" ? {
  name: "18hxk3h",
  styles: "margin-top:0;margin-bottom:0 !important"
} : {
  name: "abhq57-title",
  styles: "margin-top:0;margin-bottom:0 !important;label:title;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$a
};
var _ref5 = process.env.NODE_ENV === "production" ? {
  name: "1cl2a0e",
  styles: "display:flex;align-items:flex-start;justify-content:space-between"
} : {
  name: "m9gz0x-titleWrapper",
  styles: "display:flex;align-items:flex-start;justify-content:space-between;label:titleWrapper;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$a
};
const Header$1 = _ref6 => {
  let {
    breadcrumbs,
    title,
    titleAddOns,
    dangerouslyAppendEmotionCSS,
    buttons,
    children,
    ...rest
  } = _ref6;
  const {
    classNamePrefix: clsPrefix,
    theme
  } = useDesignSystemTheme();

  // TODO: Move to getHeaderStyles for consistency, followup ticket: https://databricks.atlassian.net/browse/FEINF-1222
  const styles = {
    titleWrapper: _ref5,
    breadcrumbWrapper: /*#__PURE__*/css({
      lineHeight: theme.typography.lineHeightBase
    }, process.env.NODE_ENV === "production" ? "" : ";label:breadcrumbWrapper;"),
    title: _ref4$1,
    // TODO: Look into a more emotion-idomatic way of doing this.
    titleIfOtherElementsPresent: _ref3$3,
    buttonContainer: _ref2$1,
    titleAddOnsWrapper: _ref$1
  };
  return jsxs("div", {
    css: [getHeaderStyles(clsPrefix, theme), dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:Header;"],
    ...rest,
    children: [breadcrumbs && jsx("div", {
      css: styles.breadcrumbWrapper,
      children: breadcrumbs
    }), jsxs("div", {
      css: styles.titleWrapper,
      children: [jsxs(Title$2, {
        level: 2,
        css: [styles.title, (buttons || breadcrumbs) && styles.titleIfOtherElementsPresent, process.env.NODE_ENV === "production" ? "" : ";label:Header;"],
        children: [title, titleAddOns && jsx("span", {
          css: styles.titleAddOnsWrapper,
          children: titleAddOns
        })]
      }), buttons && jsx("div", {
        css: styles.buttonContainer,
        children: jsx(Space, {
          size: 8,
          children: buttons
        })
      })]
    })]
  });
};

const {
  Header,
  Footer,
  Sider,
  Content: Content$2
} = Layout$1;
const Layout = /* #__PURE__ */(() => {
  const Layout = _ref => {
    let {
      children,
      dangerouslySetAntdProps,
      ...props
    } = _ref;
    return jsx(DesignSystemAntDConfigProvider, {
      children: jsx(Layout$1, {
        ...props,
        ...dangerouslySetAntdProps,
        children: jsx(RestoreAntDDefaultClsPrefix, {
          children: children
        })
      })
    });
  };
  Layout.Header = _ref2 => {
    let {
      children,
      ...props
    } = _ref2;
    return jsx(DesignSystemAntDConfigProvider, {
      children: jsx(Header, {
        ...props,
        children: jsx(RestoreAntDDefaultClsPrefix, {
          children: children
        })
      })
    });
  };
  Layout.Footer = _ref3 => {
    let {
      children,
      ...props
    } = _ref3;
    return jsx(DesignSystemAntDConfigProvider, {
      children: jsx(Footer, {
        ...props,
        children: jsx(RestoreAntDDefaultClsPrefix, {
          children: children
        })
      })
    });
  };
  Layout.Sider = /*#__PURE__*/React__default.forwardRef((_ref4, ref) => {
    let {
      children,
      ...props
    } = _ref4;
    return jsx(DesignSystemAntDConfigProvider, {
      children: jsx(Sider, {
        ...props,
        ref: ref,
        children: jsx(RestoreAntDDefaultClsPrefix, {
          children: children
        })
      })
    });
  });
  Layout.Content = _ref5 => {
    let {
      children,
      ...props
    } = _ref5;
    return jsx(DesignSystemAntDConfigProvider, {
      children: jsx(Content$2, {
        ...props,
        children: jsx(RestoreAntDDefaultClsPrefix, {
          children: children
        })
      })
    });
  };
  return Layout;
})();

function _EMOTION_STRINGIFIED_CSS_ERROR__$9() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
function getPaginationEmotionStyles(clsPrefix, theme) {
  const classRoot = `.${clsPrefix}-pagination`;
  const classItem = `.${clsPrefix}-pagination-item`;
  const classLink = `.${clsPrefix}-pagination-item-link`;
  const classActive = `.${clsPrefix}-pagination-item-active`;
  const classEllipsis = `.${clsPrefix}-pagination-item-ellipsis`;
  const classNext = `.${clsPrefix}-pagination-next`;
  const classPrev = `.${clsPrefix}-pagination-prev`;
  const classJumpNext = `.${clsPrefix}-pagination-jump-next`;
  const classJumpPrev = `.${clsPrefix}-pagination-jump-prev`;
  const classSizeChanger = `.${clsPrefix}-pagination-options-size-changer`;
  const classOptions = `.${clsPrefix}-pagination-options`;
  const styles = {
    'span[role=img]': {
      color: theme.colors.textSecondary,
      '> *': {
        color: 'inherit'
      }
    },
    [classItem]: {
      backgroundColor: 'none',
      border: 'none',
      color: theme.colors.textSecondary,
      '&:focus-visible': {
        outline: 'auto'
      },
      '> a': {
        color: theme.colors.textSecondary,
        textDecoration: 'none',
        '&:hover': {
          color: theme.colors.actionDefaultTextHover
        },
        '&:active': {
          color: theme.colors.actionDefaultTextPress
        }
      },
      '&:hover': {
        backgroundColor: theme.colors.actionDefaultBackgroundHover
      },
      '&:active': {
        backgroundColor: theme.colors.actionDefaultBackgroundPress
      }
    },
    [classActive]: {
      backgroundColor: theme.colors.actionDefaultBackgroundPress,
      color: theme.colors.actionDefaultTextPress,
      border: 'none',
      '> a': {
        color: theme.colors.actionDefaultTextPress
      },
      '&:focus-visible': {
        outline: 'auto'
      },
      '&:hover': {
        backgroundColor: theme.colors.actionDefaultBackgroundPress,
        color: theme.colors.actionDefaultTextPress
      }
    },
    [classLink]: {
      border: 'none',
      color: theme.colors.textSecondary,
      '&[disabled]': {
        display: 'none'
      },
      '&:hover': {
        backgroundColor: theme.colors.actionDefaultBackgroundHover
      },
      '&:active': {
        backgroundColor: theme.colors.actionDefaultBackgroundPress
      },
      '&:focus-visible': {
        outline: 'auto'
      }
    },
    [classEllipsis]: {
      color: 'inherit'
    },
    [`${classNext}, ${classPrev}, ${classJumpNext}, ${classJumpPrev}`]: {
      color: theme.colors.textSecondary,
      '&:hover': {
        backgroundColor: theme.colors.actionDefaultBackgroundHover
      },
      '&:active': {
        backgroundColor: theme.colors.actionDefaultBackgroundPress
      },
      '&:focus-visible': {
        outline: 'auto'
      }
    },
    [`&${classRoot}.mini, ${classRoot}.mini`]: {
      [`${classItem}, ${classNext}, ${classPrev}, ${classJumpNext}, ${classJumpPrev}`]: {
        height: '32px',
        minWidth: '32px',
        width: 'auto',
        lineHeight: '32px'
      },
      [classSizeChanger]: {
        marginLeft: 4
      },
      [`input,  ${classOptions}`]: {
        height: '32px'
      }
    }
  };
  const importantStyles = importantify(styles);
  return /*#__PURE__*/css(importantStyles, process.env.NODE_ENV === "production" ? "" : ";label:getPaginationEmotionStyles;");
}
const Pagination = function Pagination(_ref) {
  let {
    currentPageIndex,
    pageSize = 10,
    numTotal,
    onChange,
    style,
    dangerouslySetAntdProps
  } = _ref;
  const {
    classNamePrefix,
    theme
  } = useDesignSystemTheme();
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Pagination$1, {
      css: getPaginationEmotionStyles(classNamePrefix, theme),
      current: currentPageIndex,
      pageSize: pageSize,
      responsive: false,
      total: numTotal,
      onChange: onChange,
      showSizeChanger: false,
      showQuickJumper: false,
      size: 'small',
      style: style,
      ...dangerouslySetAntdProps
    })
  });
};
var _ref3$2 = process.env.NODE_ENV === "production" ? {
  name: "1u1zie3",
  styles: "width:120px"
} : {
  name: "1am9qog-CursorPagination",
  styles: "width:120px;label:CursorPagination;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$9
};
const CursorPagination = function CursorPagination(_ref2) {
  let {
    onNextPage,
    onPreviousPage,
    hasNextPage,
    hasPreviousPage,
    nextPageText = 'Next',
    previousPageText = 'Previous',
    pageSizeSelect: {
      options: pageSizeOptions,
      default: defaultPageSize,
      getOptionText: getPageSizeOptionText,
      onChange: onPageSizeChange
    } = {}
  } = _ref2;
  const {
    theme,
    classNamePrefix
  } = useDesignSystemTheme();
  const [pageSizeValue, setPageSizeValue] = useState(defaultPageSize);
  const getPageSizeOptionTextDefault = pageSize => `${pageSize} / page`;
  return jsxs("div", {
    css: /*#__PURE__*/css({
      display: 'flex',
      flexDirection: 'row',
      gap: theme.spacing.sm,
      [`.${classNamePrefix}-select-selector::after`]: {
        content: 'none'
      }
    }, process.env.NODE_ENV === "production" ? "" : ";label:CursorPagination;"),
    children: [jsx(Button, {
      icon: jsx(ChevronLeftIcon, {}),
      disabled: !hasPreviousPage,
      onClick: onPreviousPage,
      type: "tertiary",
      children: previousPageText
    }), jsx(Button, {
      endIcon: jsx(ChevronRightIcon, {}),
      disabled: !hasNextPage,
      onClick: onNextPage,
      type: "tertiary",
      children: nextPageText
    }), pageSizeOptions && jsx(Select, {
      value: String(pageSizeValue),
      css: _ref3$2,
      onChange: pageSize => {
        const updatedPageSize = Number(pageSize);
        onPageSizeChange === null || onPageSizeChange === void 0 ? void 0 : onPageSizeChange(updatedPageSize);
        setPageSizeValue(updatedPageSize);
      },
      children: pageSizeOptions.map(pageSize => jsx(Select.Option, {
        value: String(pageSize),
        children: (getPageSizeOptionText || getPageSizeOptionTextDefault)(pageSize)
      }, pageSize))
    })]
  });
};

const getTableEmotionStyles = (classNamePrefix, theme, scrollableInFlexibleContainer) => {
  const styles = [/*#__PURE__*/css({
    [`.${classNamePrefix}-table-pagination`]: {
      ...getPaginationEmotionStyles(classNamePrefix, theme)
    }
  }, process.env.NODE_ENV === "production" ? "" : ";label:styles;")];
  if (scrollableInFlexibleContainer) {
    styles.push(getScrollableInFlexibleContainerStyles(classNamePrefix));
  }
  return styles;
};
const getScrollableInFlexibleContainerStyles = clsPrefix => {
  const styles = {
    minHeight: 0,
    [`.${clsPrefix}-spin-nested-loading`]: {
      height: '100%'
    },
    [`.${clsPrefix}-spin-container`]: {
      height: '100%',
      display: 'flex',
      flexDirection: 'column'
    },
    [`.${clsPrefix}-table-container`]: {
      height: '100%',
      display: 'flex',
      flexDirection: 'column'
    },
    [`.${clsPrefix}-table`]: {
      minHeight: 0
    },
    [`.${clsPrefix}-table-header`]: {
      flexShrink: 0
    },
    [`.${clsPrefix}-table-body`]: {
      minHeight: 0
    }
  };
  return /*#__PURE__*/css(styles, process.env.NODE_ENV === "production" ? "" : ";label:getScrollableInFlexibleContainerStyles;");
};
const DEFAULT_LOADING_SPIN_PROPS = {
  indicator: jsx(Spinner, {})
};

/**
 * `LegacyTable` is deprecated in favor of the new `Table` component
 * For more information please join #dubois-table-early-adopters in Slack.
 * @deprecated
 */
// eslint-disable-next-line @typescript-eslint/ban-types
const LegacyTable = props => {
  const {
    classNamePrefix,
    theme
  } = useDesignSystemTheme();
  const {
    loading,
    scrollableInFlexibleContainer,
    children,
    ...tableProps
  } = props;
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Table$1
    // NOTE(FEINF-1273): The default loading indicator from AntD does not animate
    // and the design system spinner is recommended over the AntD one. Therefore,
    // if `loading` is `true`, render the design system <Spinner/> component.
    , {
      loading: loading === true ? DEFAULT_LOADING_SPIN_PROPS : loading,
      scroll: scrollableInFlexibleContainer ? {
        y: 'auto'
      } : undefined,
      ...tableProps,
      css: getTableEmotionStyles(classNamePrefix, theme, Boolean(scrollableInFlexibleContainer)),
      children: jsx(RestoreAntDDefaultClsPrefix, {
        children: children
      })
    })
  });
};

/**
 * @deprecated Use `DropdownMenu` instead.
 */
const Menu = /* #__PURE__ */(() => {
  const Menu = _ref => {
    let {
      dangerouslySetAntdProps,
      ...props
    } = _ref;
    return jsx(DesignSystemAntDConfigProvider, {
      children: jsx(Menu$1, {
        ...props,
        ...dangerouslySetAntdProps
      })
    });
  };
  Menu.Item = Menu$1.Item;
  Menu.ItemGroup = Menu$1.ItemGroup;
  Menu.SubMenu = function SubMenu(_ref2) {
    let {
      dangerouslySetAntdProps,
      ...props
    } = _ref2;
    const {
      theme
    } = useDesignSystemTheme();
    return jsx(ClassNames, {
      children: _ref3 => {
        let {
          css
        } = _ref3;
        return jsx(Menu$1.SubMenu, {
          popupClassName: css({
            zIndex: theme.options.zIndexBase + 50
          }),
          popupOffset: [-6, -10],
          ...props,
          ...dangerouslySetAntdProps
        });
      }
    });
  };
  return Menu;
})();

function _EMOTION_STRINGIFIED_CSS_ERROR__$8() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const SIZE_PRESETS = {
  normal: 640,
  wide: 880
};
const getModalEmotionStyles = args => {
  const {
    theme,
    clsPrefix,
    hasFooter = true,
    maxedOutHeight
  } = args;
  const classNameClose = `.${clsPrefix}-modal-close`;
  const classNameCloseX = `.${clsPrefix}-modal-close-x`;
  const classNameTitle = `.${clsPrefix}-modal-title`;
  const classNameContent = `.${clsPrefix}-modal-content`;
  const classNameBody = `.${clsPrefix}-modal-body`;
  const classNameHeader = `.${clsPrefix}-modal-header`;
  const classNameFooter = `.${clsPrefix}-modal-footer`;
  const classNameButton = `.${clsPrefix}-btn`;
  const classNameDropdownTrigger = `.${clsPrefix}-dropdown-trigger`;
  const MODAL_PADDING = theme.spacing.lg;
  const BUTTON_SIZE = theme.general.heightSm;
  // Needed for moving some of the padding from the header and footer to the content to avoid a scrollbar from appearing
  // when the content has some interior components that reach the limits of the content div
  // 8px is an arbitrary value, it still leaves enough padding for the header and footer too to avoid the same problem
  // from occurring there too
  const CONTENT_BUFFER = 8;
  const modalMaxHeight = '90vh';
  const headerHeight = 64;
  const footerHeight = hasFooter ? 52 : 0;
  const bodyMaxHeight = `calc(${modalMaxHeight} - ${headerHeight}px - ${footerHeight}px - ${MODAL_PADDING}px)`;
  return /*#__PURE__*/css({
    [classNameHeader]: {
      background: 'transparent',
      paddingTop: theme.spacing.md,
      paddingLeft: theme.spacing.lg,
      paddingRight: theme.spacing.md,
      paddingBottom: theme.spacing.md
    },
    [classNameFooter]: {
      height: footerHeight,
      paddingTop: theme.spacing.lg - CONTENT_BUFFER,
      paddingLeft: MODAL_PADDING,
      paddingRight: MODAL_PADDING,
      marginTop: 'auto',
      [`${classNameButton} + ${classNameButton}:not(${classNameDropdownTrigger})`]: {
        marginLeft: theme.spacing.sm
      }
    },
    [classNameCloseX]: {
      fontSize: theme.general.iconSize,
      height: BUTTON_SIZE,
      width: BUTTON_SIZE,
      lineHeight: 'normal',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      color: theme.colors.textSecondary
    },
    [classNameClose]: {
      height: BUTTON_SIZE,
      width: BUTTON_SIZE,
      // Note: Ant has the close button absolutely positioned, rather than in a flex container with the title.
      // This magic number is eyeballed to get the close X to align with the title text.
      margin: '16px 16px 0 0',
      borderRadius: theme.borders.borderRadiusMd,
      backgroundColor: theme.colors.actionDefaultBackgroundDefault,
      borderColor: theme.colors.actionDefaultBackgroundDefault,
      color: theme.colors.actionDefaultTextDefault,
      '&:hover': {
        backgroundColor: theme.colors.actionDefaultBackgroundHover,
        borderColor: theme.colors.actionDefaultBackgroundHover,
        color: theme.colors.actionDefaultTextHover
      },
      '&:active': {
        backgroundColor: theme.colors.actionDefaultBackgroundPress,
        borderColor: theme.colors.actionDefaultBackgroundPress,
        color: theme.colors.actionDefaultTextPress
      },
      '&:focus-visible': {
        outlineStyle: 'solid',
        outlineWidth: '2px',
        outlineOffset: '1px',
        outlineColor: theme.colors.primary
      }
    },
    [classNameTitle]: {
      fontSize: theme.typography.fontSizeXl,
      lineHeight: theme.typography.lineHeightXl,
      fontWeight: theme.typography.typographyBoldFontWeight,
      paddingRight: MODAL_PADDING,
      minHeight: headerHeight / 2,
      display: 'flex',
      alignItems: 'center'
    },
    [classNameContent]: {
      backgroundColor: theme.colors.backgroundPrimary,
      maxHeight: modalMaxHeight,
      height: maxedOutHeight ? modalMaxHeight : '',
      overflow: 'hidden',
      paddingBottom: MODAL_PADDING,
      display: 'flex',
      flexDirection: 'column',
      boxShadow: theme.general.shadowHigh
    },
    [classNameBody]: {
      overflowY: 'auto',
      maxHeight: bodyMaxHeight,
      paddingLeft: MODAL_PADDING,
      paddingRight: MODAL_PADDING,
      paddingTop: CONTENT_BUFFER,
      paddingBottom: CONTENT_BUFFER,
      ...(theme.isDarkMode === false ? {
        // Achieves an inner shadow on the content, but only when there is more left to scroll. When the content fits
        // in the container without scrolling, no shadow will be shown.
        // Taken from: https://css-tricks.com/scroll-shadows-with-javascript/
        background: `linear-gradient(
              white 30%,
              rgba(255, 255, 255, 0)
            ) center top,

            linear-gradient(
              rgba(255, 255, 255, 0),
              white 70%
            ) center bottom,

            radial-gradient(
              farthest-side at 50% 0,
              rgba(0, 0, 0, 0.2),
              rgba(0, 0, 0, 0)
            ) center top,

            radial-gradient(
              farthest-side at 50% 100%,
              rgba(0, 0, 0, 0.2),
              rgba(0, 0, 0, 0)
            ) center bottom`,
        backgroundRepeat: 'no-repeat',
        backgroundSize: '100% 40px, 100% 40px, 100% 14px, 100% 14px',
        backgroundAttachment: 'local, local, scroll, scroll'
      } : {})
    },
    ...getAnimationCss(theme.options.enableAnimation)
  }, process.env.NODE_ENV === "production" ? "" : ";label:getModalEmotionStyles;");
};

/**
 * Render default footer with our buttons. Copied from AntD.
 */
function DefaultFooter(_ref) {
  let {
    onOk,
    onCancel,
    confirmLoading,
    okText,
    cancelText,
    okButtonProps,
    cancelButtonProps,
    autoFocusButton
  } = _ref;
  const handleCancel = e => {
    onCancel === null || onCancel === void 0 ? void 0 : onCancel(e);
  };
  const handleOk = e => {
    onOk === null || onOk === void 0 ? void 0 : onOk(e);
  };
  return jsxs(Fragment, {
    children: [cancelText && jsx(Button, {
      onClick: handleCancel,
      autoFocus: autoFocusButton === 'cancel',
      dangerouslyUseFocusPseudoClass: true,
      ...cancelButtonProps,
      children: cancelText
    }), okText && jsx(Button, {
      loading: confirmLoading,
      onClick: handleOk,
      type: "primary",
      autoFocus: autoFocusButton === 'ok',
      dangerouslyUseFocusPseudoClass: true,
      ...okButtonProps,
      children: okText
    })]
  });
}
function Modal(_ref2) {
  let {
    okButtonProps,
    cancelButtonProps,
    dangerouslySetAntdProps,
    children,
    title,
    footer,
    size = 'normal',
    verticalSizing = 'dynamic',
    autoFocusButton,
    ...props
  } = _ref2;
  const {
    theme,
    classNamePrefix
  } = useDesignSystemTheme();
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Modal$1, {
      css: getModalEmotionStyles({
        theme,
        clsPrefix: classNamePrefix,
        hasFooter: footer !== null,
        maxedOutHeight: verticalSizing === 'maxed_out'
      }),
      title: jsx(RestoreAntDDefaultClsPrefix, {
        children: title
      }),
      footer: jsx(RestoreAntDDefaultClsPrefix, {
        children: footer === undefined ? jsx(DefaultFooter, {
          onOk: props.onOk,
          onCancel: props.onCancel,
          confirmLoading: props.confirmLoading,
          okText: props.okText,
          cancelText: props.cancelText,
          okButtonProps: okButtonProps,
          cancelButtonProps: cancelButtonProps,
          autoFocusButton: autoFocusButton
        }) : footer
      }),
      width: size ? SIZE_PRESETS[size] : undefined,
      closeIcon: jsx(CloseIcon, {}),
      centered: true,
      zIndex: theme.options.zIndexBase,
      ...props,
      ...dangerouslySetAntdProps,
      children: jsx(RestoreAntDDefaultClsPrefix, {
        children: children
      })
    })
  });
}
var _ref3$1 = process.env.NODE_ENV === "production" ? {
  name: "b9hrb",
  styles: "position:relative;display:inline-flex;align-items:center"
} : {
  name: "1jkwrsj-titleComp",
  styles: "position:relative;display:inline-flex;align-items:center;label:titleComp;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$8
};
var _ref4 = process.env.NODE_ENV === "production" ? {
  name: "1o6wc9k",
  styles: "padding-left:6px"
} : {
  name: "i303lp-titleComp",
  styles: "padding-left:6px;label:titleComp;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$8
};
function DangerModal(props) {
  const {
    theme
  } = useDesignSystemTheme();
  const {
    USE_NEW_ICONS: useNewIcons
  } = useDesignSystemFlags();
  const {
    title,
    onCancel,
    onOk,
    cancelText,
    okText,
    okButtonProps,
    cancelButtonProps,
    ...restProps
  } = props;
  const iconSize = useNewIcons ? 18 : 20;
  const iconFontSize = useNewIcons ? 18 : 22;
  const titleComp = jsxs("div", {
    css: _ref3$1,
    children: [jsx(DangerIcon, {
      css: /*#__PURE__*/css({
        color: theme.colors.textValidationDanger,
        left: 2,
        height: iconSize,
        width: iconSize,
        fontSize: iconFontSize
      }, process.env.NODE_ENV === "production" ? "" : ";label:titleComp;")
    }), jsx("div", {
      css: _ref4,
      children: title
    })]
  });
  return jsx(Modal, {
    title: titleComp,
    footer: [jsx(Button, {
      onClick: onCancel,
      ...cancelButtonProps,
      children: cancelText || 'Cancel'
    }, "cancel"), jsx(Button, {
      type: "primary",
      danger: true,
      onClick: onOk,
      loading: props.confirmLoading,
      ...okButtonProps,
      children: okText || 'Delete'
    }, "discard")],
    onOk: onOk,
    onCancel: onCancel,
    ...restProps
  });
}

// Note: AntD only exposes context to notifications via the `useNotification` hook, and we need context to apply themes
// to AntD. As such you can currently only use notifications from within functional components.
function useNotification() {
  const [notificationInstance, contextHolder] = notification.useNotification();
  const {
    getPrefixedClassName,
    theme
  } = useDesignSystemTheme();
  const {
    getPopupContainer: getContainer
  } = useDesignSystemContext();
  const clsPrefix = getPrefixedClassName('notification');
  const open = useCallback(args => {
    const mergedArgs = {
      getContainer,
      ...defaultProps,
      ...args,
      style: {
        zIndex: theme.options.zIndexBase + 30,
        boxShadow: theme.general.shadowLow
      }
    };
    const iconClassName = `${clsPrefix}-notice-icon-${mergedArgs.type}`;
    mergedArgs.icon = jsx(SeverityIcon, {
      severity: mergedArgs.type,
      className: iconClassName
    });
    mergedArgs.closeIcon = jsx(CloseIcon, {
      css: /*#__PURE__*/css({
        cursor: 'pointer',
        fontSize: theme.general.iconSize
      }, process.env.NODE_ENV === "production" ? "" : ";label:mergedArgs-closeIcon;"),
      "aria-label": mergedArgs.closeLabel || 'Close notification'
    });
    notificationInstance.open(mergedArgs);
  }, [notificationInstance, getContainer, theme, clsPrefix]);
  const wrappedNotificationAPI = useMemo(() => {
    const error = args => open({
      ...args,
      type: 'error'
    });
    const warning = args => open({
      ...args,
      type: 'warning'
    });
    const info = args => open({
      ...args,
      type: 'info'
    });
    const success = args => open({
      ...args,
      type: 'success'
    });
    const close = key => notification.close(key);
    return {
      open,
      close,
      error,
      warning,
      info,
      success
    };
  }, [open]);

  // eslint-disable-next-line react/jsx-key -- TODO(FEINF-1756)
  return [wrappedNotificationAPI, jsx(DesignSystemAntDConfigProvider, {
    children: contextHolder
  })];
}
const defaultProps = {
  type: 'info',
  duration: 3
};

/**
 * A type wrapping given component interface with props returned by withNotifications() HOC
 *
 * @deprecated Please migrate components to functional components and use useNotification() hook instead.
 */

/**
 * A higher-order component factory function, enables using notifications in
 * class components in a similar way to useNotification() hook. Wrapped component will have
 * additional "notificationAPI" and "notificationContextHolder" props injected containing
 * the notification API object and context holder react node respectively.
 *
 * The wrapped component can implement WithNotificationsHOCProps<OwnProps> type which
 * enriches the component's interface with the mentioned props.
 *
 * @deprecated Please migrate components to functional components and use useNotification() hook instead.
 */
const withNotifications = Component => /*#__PURE__*/forwardRef((props, ref) => {
  const [notificationAPI, notificationContextHolder] = useNotification();
  return jsx(Component, {
    ref: ref,
    notificationAPI: notificationAPI,
    notificationContextHolder: notificationContextHolder,
    ...props
  });
});

const hideAnimation = keyframes({
  from: {
    opacity: 1
  },
  to: {
    opacity: 0
  }
});
const slideInAnimation = keyframes({
  from: {
    transform: 'translateX(calc(100% + 12px))'
  },
  to: {
    transform: 'translateX(0)'
  }
});
const swipeOutAnimation = keyframes({
  from: {
    transform: 'translateX(var(--radix-toast-swipe-end-x))'
  },
  to: {
    transform: 'translateX(calc(100% + 12px))'
  }
});
const getToastRootStyle = (theme, classNamePrefix) => {
  return /*#__PURE__*/css({
    '&&': {
      position: 'relative',
      display: 'grid',
      background: theme.colors.backgroundPrimary,
      padding: 12,
      columnGap: 4,
      boxShadow: theme.general.shadowLow,
      borderRadius: theme.general.borderRadiusBase,
      lineHeight: '20px',
      gridTemplateRows: '[header] auto [content] auto',
      gridTemplateColumns: '[icon] auto [content] 1fr [close] auto'
    },
    [`.${classNamePrefix}-notification-severity-icon`]: {
      gridRow: 'header / content',
      gridColumn: 'icon / icon',
      display: 'inline-flex',
      alignItems: 'center'
    },
    [`.${classNamePrefix}-btn`]: {
      display: 'inline-flex',
      alignItems: 'center',
      justifyContent: 'center'
    },
    [`.${classNamePrefix}-notification-info-icon`]: {
      color: theme.colors.textSecondary
    },
    [`.${classNamePrefix}-notification-success-icon`]: {
      color: theme.colors.textValidationSuccess
    },
    [`.${classNamePrefix}-notification-warning-icon`]: {
      color: theme.colors.textValidationWarning
    },
    [`.${classNamePrefix}-notification-error-icon`]: {
      color: theme.colors.textValidationDanger
    },
    '&&[data-state="open"]': {
      animation: `${slideInAnimation} 300ms cubic-bezier(0.16, 1, 0.3, 1)`
    },
    '&[data-state="closed"]': {
      animation: `${hideAnimation} 100ms ease-in`
    },
    '&[data-swipe="move"]': {
      transform: 'translateX(var(--radix-toast-swipe-move-x))'
    },
    '&[data-swipe="cancel"]': {
      transform: 'translateX(0)',
      transition: 'transform 200ms ease-out'
    },
    '&[data-swipe="end"]': {
      animation: `${swipeOutAnimation} 100ms ease-out`
    }
  }, process.env.NODE_ENV === "production" ? "" : ";label:getToastRootStyle;");
};
const Root$1 = /*#__PURE__*/forwardRef(function (_ref, ref) {
  let {
    children,
    severity = 'info',
    ...props
  } = _ref;
  const {
    theme,
    classNamePrefix
  } = useDesignSystemTheme();
  return jsxs(Toast.Root, {
    ref: ref,
    css: getToastRootStyle(theme, classNamePrefix),
    ...props,
    children: [jsx(SeverityIcon, {
      className: `${classNamePrefix}-notification-severity-icon ${classNamePrefix}-notification-${severity}-icon`,
      severity: severity
    }), children]
  });
});

// TODO: Support light and dark mode

const getViewportStyle = theme => {
  return {
    position: 'fixed',
    top: 0,
    right: 0,
    display: 'flex',
    flexDirection: 'column',
    padding: 12,
    gap: 12,
    width: 440,
    listStyle: 'none',
    zIndex: theme.options.zIndexBase + 100,
    outline: 'none'
  };
};
const getTitleStyles = theme => {
  return /*#__PURE__*/css({
    fontWeight: theme.typography.typographyBoldFontWeight,
    color: theme.colors.textPrimary,
    gridRow: 'header / header',
    gridColumn: 'content / content'
  }, process.env.NODE_ENV === "production" ? "" : ";label:getTitleStyles;");
};
const Title = /*#__PURE__*/forwardRef(function (_ref2, ref) {
  let {
    children,
    ...props
  } = _ref2;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(Toast.Title, {
    ref: ref,
    css: getTitleStyles(theme),
    ...props,
    children: children
  });
});
const getDescriptionStyles = theme => {
  return /*#__PURE__*/css({
    marginTop: 4,
    color: theme.colors.textPrimary,
    gridRow: 'content / content',
    gridColumn: 'content / content'
  }, process.env.NODE_ENV === "production" ? "" : ";label:getDescriptionStyles;");
};
const Description = /*#__PURE__*/forwardRef(function (_ref3, ref) {
  let {
    children,
    ...props
  } = _ref3;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(Toast.Description, {
    ref: ref,
    css: getDescriptionStyles(theme),
    ...props,
    children: children
  });
});
const getCloseStyles = theme => {
  return /*#__PURE__*/css({
    color: theme.colors.textSecondary,
    position: 'absolute',
    // Offset close button position to align with the title, title uses 20px line height, button has 32px
    right: 6,
    top: 6
  }, process.env.NODE_ENV === "production" ? "" : ";label:getCloseStyles;");
};
const Close$1 = /*#__PURE__*/forwardRef(function (props, ref) {
  var _ref4;
  const {
    theme
  } = useDesignSystemTheme();
  const {
    closeLabel,
    ...restProps
  } = props;
  return (
    // Wrapper to keep close column width for content sizing, close button positioned absolute for alignment without affecting the grid's first row height (title)
    jsx("div", {
      style: {
        gridColumn: 'close / close',
        gridRow: 'header / content',
        width: 20
      },
      children: jsx(Toast.Close, {
        ref: ref,
        css: getCloseStyles(theme),
        ...restProps,
        asChild: true,
        children: jsx(Button, {
          icon: jsx(CloseIcon, {}),
          "aria-label": (_ref4 = closeLabel !== null && closeLabel !== void 0 ? closeLabel : restProps['aria-label']) !== null && _ref4 !== void 0 ? _ref4 : 'Close notification'
        })
      })
    })
  );
});
const Provider = _ref5 => {
  let {
    children,
    ...props
  } = _ref5;
  return jsx(Toast.Provider, {
    ...props,
    children: children
  });
};
const Viewport = props => {
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(Toast.Viewport, {
    className: DU_BOIS_ENABLE_ANIMATION_CLASSNAME,
    style: getViewportStyle(theme),
    ...props
  });
};

var NotificationV2 = /*#__PURE__*/Object.freeze({
  __proto__: null,
  Root: Root$1,
  Title: Title,
  Description: Description,
  Close: Close$1,
  Provider: Provider,
  Viewport: Viewport
});

const PageWrapper = _ref => {
  let {
    children,
    ...props
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx("div", {
    css: /*#__PURE__*/css({
      paddingLeft: 16,
      paddingRight: 16,
      backgroundColor: theme.isDarkMode ? theme.colors.backgroundPrimary : 'transparent'
    }, process.env.NODE_ENV === "production" ? "" : ";label:PageWrapper;"),
    ...props,
    children: children
  });
};

const Popover = _ref => {
  let {
    content,
    dangerouslySetAntdProps,
    ...props
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Popover$2, {
      zIndex: theme.options.zIndexBase + 30,
      ...props,
      content: jsx(RestoreAntDDefaultClsPrefix, {
        children: content
      })
    })
  });
};

const Root = Popover$1.Root; // Behavioral component only

const Content$1 = /*#__PURE__*/forwardRef(function Content(_ref, ref) {
  let {
    children,
    minWidth = 220,
    ...props
  } = _ref;
  const {
    getPopupContainer
  } = useDesignSystemContext();
  return jsx(Popover$1.Portal, {
    container: getPopupContainer && getPopupContainer(),
    children: jsx(Popover$1.Content, {
      ref: ref,
      css: [contentStyles, {
        minWidth
      }, process.env.NODE_ENV === "production" ? "" : ";label:Content;"],
      sideOffset: 4,
      ...props,
      children: children
    })
  });
});
const Trigger = /*#__PURE__*/forwardRef(function Trigger(_ref2, ref) {
  let {
    children,
    ...props
  } = _ref2;
  return jsx(Popover$1.Trigger, {
    ref: ref,
    ...props,
    children: children
  });
});
const Close = /*#__PURE__*/forwardRef(function Close(_ref3, ref) {
  let {
    children,
    ...props
  } = _ref3;
  return jsx(Popover$1.Close, {
    ref: ref,
    ...props,
    children: children
  });
});
const Arrow = /*#__PURE__*/forwardRef(function Arrow(_ref4, ref) {
  let {
    children,
    ...props
  } = _ref4;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(Popover$1.Arrow, {
    css: /*#__PURE__*/css({
      fill: theme.colors.backgroundPrimary,
      stroke: theme.colors.borderDecorative,
      strokeDashoffset: -CONSTANTS.arrowBottomLength(),
      strokeDasharray: CONSTANTS.arrowBottomLength() + 2 * CONSTANTS.arrowSide(),
      strokeWidth: CONSTANTS.arrowStrokeWidth(),
      // TODO: This is a temporary fix for the alignment of the Arrow;
      // Radix has changed the implementation for v1.0.0 (uses floating-ui)
      // which has new behaviors for alignment that we don't want. Generally
      // we need to fix the arrow to always be aligned to the left of the menu (with
      // offset equal to border radius)
      position: 'relative',
      top: -1
    }, process.env.NODE_ENV === "production" ? "" : ";label:Arrow;"),
    ref: ref,
    width: 12,
    height: 6,
    ...props,
    children: children
  });
});

// CONSTANTS
const CONSTANTS = {
  arrowBottomLength() {
    // The built in arrow is a polygon: 0,0 30,0 15,10
    return 30;
  },
  arrowHeight() {
    return 10;
  },
  arrowSide() {
    return 2 * (this.arrowHeight() ** 2 * 2) ** 0.5;
  },
  arrowStrokeWidth() {
    // This is eyeballed b/c relative to the svg viewbox coordinate system
    return 2;
  }
};
const popoverContentStyles = theme => ({
  backgroundColor: theme.colors.backgroundPrimary,
  color: theme.colors.textPrimary,
  lineHeight: theme.typography.lineHeightBase,
  border: `1px solid ${theme.colors.borderDecorative}`,
  borderRadius: theme.borders.borderRadiusMd,
  padding: `${theme.spacing.sm}px`,
  boxShadow: theme.general.shadowLow,
  userSelect: 'none',
  zIndex: theme.options.zIndexBase + 30,
  a: importantify({
    color: theme.colors.actionTertiaryTextDefault,
    cursor: 'default',
    '&:hover, &:focus': {
      color: theme.colors.actionTertiaryTextHover
    }
  }),
  '&:focus-visible': {
    outlineStyle: 'solid',
    outlineWidth: '2px',
    outlineOffset: '1px',
    outlineColor: theme.colors.primary
  }
});
const contentStyles = theme => ({
  ...popoverContentStyles(theme)
});

var PopoverV2 = /*#__PURE__*/Object.freeze({
  __proto__: null,
  Root: Root,
  Content: Content$1,
  Trigger: Trigger,
  Close: Close,
  Arrow: Arrow
});

const SMALL_BUTTON_HEIGHT = 24;
function getSegmentedControlGroupEmotionStyles(clsPrefix) {
  const classSmallGroup = `.${clsPrefix}-radio-group-small`;
  const classButtonWrapper = `.${clsPrefix}-radio-button-wrapper`;
  const styles = {
    [`&${classSmallGroup} ${classButtonWrapper}`]: {
      padding: '0 12px'
    }
  };
  const importantStyles = importantify(styles);
  return /*#__PURE__*/css(importantStyles, process.env.NODE_ENV === "production" ? "" : ";label:getSegmentedControlGroupEmotionStyles;");
}
function getSegmentedControlButtonEmotionStyles(clsPrefix, theme, size) {
  const classWrapperChecked = `.${clsPrefix}-radio-button-wrapper-checked`;
  const classWrapper = `.${clsPrefix}-radio-button-wrapper`;
  const classWrapperDisabled = `.${clsPrefix}-radio-button-wrapper-disabled`;
  const classButton = `.${clsPrefix}-radio-button`;

  // Note: Ant radio button uses a 1px-wide `before` pseudo-element to recreate the left border of the button.
  // This is because the actual left border is disabled to avoid a double-border effect with the adjacent button's
  // right border.
  // We must override the background colour of this pseudo-border to be the same as the real border above.

  const styles = {
    backgroundColor: theme.colors.actionDefaultBackgroundDefault,
    borderColor: theme.colors.actionDefaultBorderDefault,
    color: theme.colors.actionDefaultTextDefault,
    '::before': {
      backgroundColor: theme.colors.actionDefaultBorderDefault
    },
    '&:hover': {
      backgroundColor: theme.colors.actionDefaultBackgroundHover,
      borderColor: theme.colors.actionDefaultBorderHover,
      color: theme.colors.actionDefaultTextHover,
      '::before': {
        backgroundColor: theme.colors.actionDefaultBorderHover
      },
      // Also target the same pseudo-element on the next sibling, because this is used to create the right border
      [`& + ${classWrapper}::before`]: {
        backgroundColor: theme.colors.actionDefaultBorderPress
      }
    },
    '&:active': {
      backgroundColor: theme.colors.actionTertiaryBackgroundPress,
      borderColor: theme.colors.actionDefaultBorderPress,
      color: theme.colors.actionTertiaryTextPress
    },
    [`&${classWrapperChecked}`]: {
      backgroundColor: theme.colors.actionTertiaryBackgroundPress,
      borderColor: theme.colors.actionDefaultBorderPress,
      color: theme.colors.actionTertiaryTextPress,
      boxShadow: 'none',
      '::before': {
        backgroundColor: theme.colors.actionDefaultBorderPress
      }
    },
    [`&${classWrapperChecked}:focus-within`]: {
      '::before': {
        width: 0
      }
    },
    [`&${classWrapperDisabled}`]: {
      color: theme.colors.actionDisabledText,
      backgroundColor: theme.colors.actionDisabledBackground,
      '&:hover': {
        color: theme.colors.actionDisabledText,
        backgroundColor: theme.colors.actionDisabledBackground
      },
      '&:active': {
        color: theme.colors.actionDisabledText,
        backgroundColor: theme.colors.actionDisabledBackground
      }
    },
    [`&${classWrapper}`]: {
      padding: size === 'middle' ? '0 16px' : '0 8px',
      display: 'inline-flex',
      verticalAlign: 'middle',
      '&:focus-within': {
        outlineStyle: 'solid',
        outlineWidth: '2px',
        outlineOffset: '-2px',
        outlineColor: theme.colors.primary
      }
    },
    [`&${classWrapper}, ${classButton}`]: {
      height: size === 'middle' ? theme.general.heightSm : SMALL_BUTTON_HEIGHT,
      lineHeight: theme.typography.lineHeightBase,
      alignItems: 'center'
    },
    ...getAnimationCss(theme.options.enableAnimation)
  };
  const importantStyles = importantify(styles);
  return /*#__PURE__*/css(importantStyles, process.env.NODE_ENV === "production" ? "" : ";label:getSegmentedControlButtonEmotionStyles;");
}
const SegmentedControlGroupContext = /*#__PURE__*/createContext({
  size: 'middle'
});
const SegmentedControlButton = /*#__PURE__*/forwardRef(function SegmentedControlButton(_ref, ref) {
  let {
    dangerouslySetAntdProps,
    ...props
  } = _ref;
  const {
    classNamePrefix,
    theme
  } = useDesignSystemTheme();
  const {
    size
  } = useContext(SegmentedControlGroupContext);
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Radio$1.Button, {
      css: getSegmentedControlButtonEmotionStyles(classNamePrefix, theme, size),
      ...props,
      ...dangerouslySetAntdProps,
      ref: ref
    })
  });
});
const SegmentedControlGroup = /*#__PURE__*/forwardRef(function SegmentedControlGroup(_ref2, ref) {
  let {
    dangerouslySetAntdProps,
    size = 'middle',
    ...props
  } = _ref2;
  const {
    classNamePrefix
  } = useDesignSystemTheme();
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(SegmentedControlGroupContext.Provider, {
      value: {
        size
      },
      children: jsx(Radio$1.Group, {
        ...props,
        css: getSegmentedControlGroupEmotionStyles(classNamePrefix),
        ...dangerouslySetAntdProps,
        ref: ref
      })
    })
  });
});

function _EMOTION_STRINGIFIED_CSS_ERROR__$7() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const DEFAULT_WIDTH = 200;
const ContentContextDefaults = {
  openPanelId: undefined,
  closable: true,
  destroyInactivePanels: false,
  setIsClosed: () => {}
};
const SidebarContextDefaults = {
  position: 'left'
};
const ContentContext = /*#__PURE__*/createContext(ContentContextDefaults);
const SidebarContext = /*#__PURE__*/createContext(SidebarContextDefaults);
function Nav(_ref) {
  let {
    children,
    dangerouslyAppendEmotionCSS
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx("nav", {
    css: [{
      display: 'flex',
      flexDirection: 'column',
      gap: theme.spacing.xs,
      padding: theme.spacing.xs
    }, dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:Nav;"],
    children: children
  });
}
function NavButton(_ref2) {
  let {
    active,
    disabled,
    icon,
    onClick,
    children,
    dangerouslyAppendEmotionCSS
  } = _ref2;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx("div", {
    css: [active ? importantify({
      borderRadius: theme.borders.borderRadiusMd,
      background: theme.colors.actionDefaultBackgroundPress,
      button: {
        '&:enabled:not(:hover):not(:active) > .anticon': {
          color: theme.colors.actionTertiaryTextPress
        }
      }
    }) : undefined, dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:NavButton;"],
    children: jsx(Button, {
      icon: icon,
      onClick: onClick,
      disabled: disabled,
      children: children
    })
  });
}
function Content(_ref3) {
  let {
    disableResize,
    openPanelId,
    closable,
    onClose,
    onResizeStop,
    width,
    minWidth,
    maxWidth,
    destroyInactivePanels = false,
    children,
    dangerouslyAppendEmotionCSS
  } = _ref3;
  const openAnimation = keyframes`
  from { width: 50px }
  to   { width: ${width}px }`;
  const showAnimation = keyframes`
  from { opacity: 0 }
  80%  { opacity: 0 }
  to   { opacity: 1 }`;
  const {
    theme
  } = useDesignSystemTheme();
  const sidebarContext = useContext(SidebarContext);
  const onCloseRef = useRef(onClose);
  const resizeHandleStyle = sidebarContext.position === 'right' ? {
    left: 0
  } : {
    right: 0
  };
  const [dragging, setDragging] = useState(false);
  const isPanelClosed = openPanelId == null;
  const [animation, setAnimation] = useState(isPanelClosed ? {
    open: `${openAnimation} .2s cubic-bezier(0, 0, 0.2, 1)`,
    show: `${showAnimation} .25s linear`
  } : undefined);
  const hiddenPanelStyle = /*#__PURE__*/css(isPanelClosed && {
    display: 'none'
  }, process.env.NODE_ENV === "production" ? "" : ";label:hiddenPanelStyle;");
  const containerStyle = /*#__PURE__*/css({
    animation: animation === null || animation === void 0 ? void 0 : animation.open,
    direction: sidebarContext.position === 'right' ? 'rtl' : 'ltr',
    marginLeft: -1,
    position: 'relative',
    borderWidth: sidebarContext.position === 'right' ? `0 ${theme.general.borderWidth}px 0 0 ` : `0 0 0 ${theme.general.borderWidth}px`,
    borderStyle: 'inherit',
    borderColor: 'inherit',
    boxSizing: 'content-box'
  }, process.env.NODE_ENV === "production" ? "" : ";label:containerStyle;");
  const highlightedBorderStyle = sidebarContext.position === 'right' ? /*#__PURE__*/css({
    borderLeft: `2px solid ${theme.colors.actionDefaultBorderHover}`
  }, process.env.NODE_ENV === "production" ? "" : ";label:highlightedBorderStyle;") : /*#__PURE__*/css({
    borderRight: `2px solid ${theme.colors.actionDefaultBorderHover}`
  }, process.env.NODE_ENV === "production" ? "" : ";label:highlightedBorderStyle;");
  useEffect(() => {
    onCloseRef.current = onClose;
  }, [onClose]);
  const value = useMemo(() => ({
    openPanelId,
    closable: closable === undefined ? true : closable,
    destroyInactivePanels,
    setIsClosed: () => {
      var _onCloseRef$current;
      (_onCloseRef$current = onCloseRef.current) === null || _onCloseRef$current === void 0 ? void 0 : _onCloseRef$current.call(onCloseRef);
      if (!animation) {
        setAnimation({
          open: `${openAnimation} .2s cubic-bezier(0, 0, 0.2, 1)`,
          show: `${showAnimation} .25s linear`
        });
      }
    }
  }), [openPanelId, closable, openAnimation, showAnimation, animation, destroyInactivePanels]);
  return jsx(ContentContext.Provider, {
    value: value,
    children: disableResize ? jsx("div", {
      css: [/*#__PURE__*/css(containerStyle, {
        width: width || '100%',
        height: '100%'
      }, process.env.NODE_ENV === "production" ? "" : ";label:Content;"), dangerouslyAppendEmotionCSS, hiddenPanelStyle, process.env.NODE_ENV === "production" ? "" : ";label:Content;"],
      "aria-hidden": isPanelClosed,
      children: jsx("div", {
        css: /*#__PURE__*/css({
          opacity: 1,
          height: '100%',
          animation: animation === null || animation === void 0 ? void 0 : animation.show,
          direction: 'ltr'
        }, process.env.NODE_ENV === "production" ? "" : ";label:Content;"),
        children: children
      })
    }) : jsx(ResizableBox, {
      width: width || DEFAULT_WIDTH,
      height: undefined,
      axis: "x",
      resizeHandles: sidebarContext.position === 'right' ? ['w'] : ['e'],
      minConstraints: [minWidth !== null && minWidth !== void 0 ? minWidth : DEFAULT_WIDTH, 150],
      maxConstraints: [maxWidth !== null && maxWidth !== void 0 ? maxWidth : 800, 150],
      onResizeStart: () => setDragging(true),
      onResizeStop: (_, _ref4) => {
        let {
          size
        } = _ref4;
        onResizeStop === null || onResizeStop === void 0 ? void 0 : onResizeStop(size.width);
        setDragging(false);
      },
      handle: jsx("div", {
        css: /*#__PURE__*/css({
          width: 10,
          height: '100%',
          position: 'absolute',
          top: 0,
          cursor: sidebarContext.position === 'right' ? 'w-resize' : 'e-resize',
          '&:hover': highlightedBorderStyle,
          ...resizeHandleStyle
        }, dragging && highlightedBorderStyle, process.env.NODE_ENV === "production" ? "" : ";label:Content;")
      }),
      css: [containerStyle, hiddenPanelStyle, process.env.NODE_ENV === "production" ? "" : ";label:Content;"],
      "aria-hidden": isPanelClosed,
      children: jsx("div", {
        css: [{
          opacity: 1,
          animation: animation === null || animation === void 0 ? void 0 : animation.show,
          direction: 'ltr',
          height: '100%'
        }, dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:Content;"],
        children: children
      })
    })
  });
}
function Panel(_ref5) {
  let {
    panelId,
    children,
    forceRender = false,
    dangerouslyAppendEmotionCSS,
    ...delegated
  } = _ref5;
  const {
    openPanelId,
    destroyInactivePanels
  } = useContext(ContentContext);
  const hasOpenedPanelRef = useRef(false);
  const isPanelOpen = openPanelId === panelId;
  if (isPanelOpen && !hasOpenedPanelRef.current) {
    hasOpenedPanelRef.current = true;
  }
  if ((destroyInactivePanels || !hasOpenedPanelRef.current) && !isPanelOpen && !forceRender) return null;
  return jsx("div", {
    css: ["display:flex;height:100%;flex-direction:column;", dangerouslyAppendEmotionCSS, !isPanelOpen && {
      display: 'none'
    }, process.env.NODE_ENV === "production" ? "" : ";label:Panel;"],
    "aria-hidden": !isPanelOpen,
    ...delegated,
    children: children
  });
}
var _ref7 = process.env.NODE_ENV === "production" ? {
  name: "1066lcq",
  styles: "display:flex;justify-content:space-between;align-items:center"
} : {
  name: "fs19p8-PanelHeader",
  styles: "display:flex;justify-content:space-between;align-items:center;label:PanelHeader;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$7
};
function PanelHeader(_ref6) {
  let {
    children,
    dangerouslyAppendEmotionCSS
  } = _ref6;
  const {
    theme
  } = useDesignSystemTheme();
  const contentContext = useContext(ContentContext);
  return jsxs("div", {
    css: [{
      display: 'flex',
      paddingLeft: 8,
      paddingRight: 4,
      alignItems: 'center',
      minHeight: theme.general.heightSm,
      justifyContent: 'space-between',
      fontWeight: theme.typography.typographyBoldFontWeight,
      color: theme.colors.textPrimary
    }, dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:PanelHeader;"],
    children: [jsx("div", {
      css: /*#__PURE__*/css({
        width: contentContext.closable ? `calc(100% - ${theme.spacing.lg}px)` : '100%'
      }, process.env.NODE_ENV === "production" ? "" : ";label:PanelHeader;"),
      children: jsx("div", {
        css: _ref7,
        children: children
      })
    }), contentContext.closable ? jsx("div", {
      children: jsx(Button, {
        size: "small",
        icon: jsx(CloseIcon, {}),
        "aria-label": "Close",
        onClick: () => {
          contentContext.setIsClosed();
        }
      })
    }) : null]
  });
}
function PanelHeaderTitle(_ref8) {
  let {
    title,
    dangerouslyAppendEmotionCSS
  } = _ref8;
  return jsx("div", {
    title: title,
    css: ["align-self:center;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;", dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:PanelHeaderTitle;"],
    children: title
  });
}
function PanelHeaderButtons(_ref9) {
  let {
    children,
    dangerouslyAppendEmotionCSS
  } = _ref9;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx("div", {
    css: [{
      display: 'flex',
      alignItems: 'center',
      gap: theme.spacing.xs,
      paddingRight: theme.spacing.xs
    }, dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:PanelHeaderButtons;"],
    children: children
  });
}
function PanelBody(_ref10) {
  let {
    children,
    dangerouslyAppendEmotionCSS
  } = _ref10;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx("div", {
    css: [{
      height: '100%',
      overflowX: 'hidden',
      overflowY: 'auto',
      padding: '0 8px',
      colorScheme: theme.isDarkMode ? 'dark' : 'light'
    }, dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:PanelBody;"],
    children: children
  });
}
const Sidebar = /* #__PURE__ */(() => {
  function Sidebar(_ref11) {
    let {
      position,
      children,
      dangerouslyAppendEmotionCSS
    } = _ref11;
    const {
      theme
    } = useDesignSystemTheme();
    const value = useMemo(() => {
      return {
        position: position || 'left'
      };
    }, [position]);
    return jsx(SidebarContext.Provider, {
      value: value,
      children: jsx("div", {
        css: [{
          display: 'flex',
          height: '100%',
          backgroundColor: theme.colors.backgroundPrimary,
          flexDirection: position === 'right' ? 'row-reverse' : 'row',
          borderStyle: 'solid',
          borderColor: theme.colors.borderDecorative,
          borderWidth: theme.general.borderWidth,
          boxSizing: 'content-box'
        }, dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:Sidebar;"],
        children: children
      })
    });
  }
  Sidebar.Content = Content;
  Sidebar.Nav = Nav;
  Sidebar.NavButton = NavButton;
  Sidebar.Panel = Panel;
  Sidebar.PanelHeader = PanelHeader;
  Sidebar.PanelHeaderTitle = PanelHeaderTitle;
  Sidebar.PanelHeaderButtons = PanelHeaderButtons;
  Sidebar.PanelBody = PanelBody;
  return Sidebar;
})();

/**
 * Used to hide text visually, but still be readable by screen-readers
 * and other assistive devices.
 *
 * https://www.tpgi.com/the-anatomy-of-visually-hidden/
 */
const visuallyHidden = {
  '&:not(:focus):not(:active)': {
    clip: 'rect(0 0 0 0)',
    clipPath: 'inset(50%)',
    height: '1px',
    overflow: 'hidden',
    position: 'absolute',
    whiteSpace: 'nowrap',
    width: '1px'
  }
};

function _EMOTION_STRINGIFIED_CSS_ERROR__$6() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const Skeleton = /* #__PURE__ */(() => {
  const Skeleton = _ref => {
    let {
      dangerouslySetAntdProps,
      label,
      ...props
    } = _ref;
    return jsx(DesignSystemAntDConfigProvider, {
      children: jsx(AccessibleContainer, {
        label: label,
        children: jsx(Skeleton$1, {
          ...props,
          ...dangerouslySetAntdProps
        })
      })
    });
  };
  Skeleton.Button = Skeleton$1.Button;
  Skeleton.Image = Skeleton$1.Image;
  Skeleton.Input = Skeleton$1.Input;
  return Skeleton;
})();
var _ref3 = process.env.NODE_ENV === "production" ? {
  name: "g8zzui",
  styles: "cursor:progress"
} : {
  name: "4y1qki-AccessibleContainer",
  styles: "cursor:progress;label:AccessibleContainer;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$6
};
function AccessibleContainer(_ref2) {
  let {
    children,
    label
  } = _ref2;
  if (!label) {
    return jsx(Fragment, {
      children: children
    });
  }
  return jsxs("div", {
    css: _ref3,
    children: [jsx("span", {
      css: visuallyHidden,
      children: label
    }), jsx("div", {
      "aria-hidden": true,
      children: children
    })]
  });
}

const defaultGetPrefixCls = (suffixCls, customizePrefixCls) => {
  if (customizePrefixCls) return customizePrefixCls;
  return suffixCls ? `ant-${suffixCls}` : 'ant';
};
const ConfigContext = /*#__PURE__*/React.createContext({
  // We provide a default function for Context without provider
  getPrefixCls: defaultGetPrefixCls
});
const ButtonGroup = Button$1.Group;
const DropdownButton = props => {
  const {
    getPopupContainer: getContextPopupContainer,
    getPrefixCls,
    direction
  } = React.useContext(ConfigContext);
  const {
    prefixCls: customizePrefixCls,
    type,
    danger,
    disabled,
    loading,
    onClick,
    htmlType,
    children,
    className,
    overlay,
    trigger,
    align,
    open,
    onOpenChange,
    placement,
    getPopupContainer,
    href,
    icon = jsx(AntDIcon, {}),
    title,
    buttonsRender = buttons => buttons,
    mouseEnterDelay,
    mouseLeaveDelay,
    overlayClassName,
    overlayStyle,
    destroyPopupOnHide,
    menuButtonLabel = 'Open dropdown',
    menu,
    leftButtonIcon,
    dropdownMenuRootProps,
    ...restProps
  } = props;
  const prefixCls = getPrefixCls('dropdown-button', customizePrefixCls);
  const dropdownProps = {
    align,
    overlay,
    disabled,
    trigger: disabled ? [] : trigger,
    onOpenChange,
    getPopupContainer: getPopupContainer || getContextPopupContainer,
    mouseEnterDelay,
    mouseLeaveDelay,
    overlayClassName,
    overlayStyle,
    destroyPopupOnHide
  };
  if ('open' in props) {
    dropdownProps.open = open;
  }
  if ('placement' in props) {
    dropdownProps.placement = placement;
  } else {
    dropdownProps.placement = direction === 'rtl' ? 'bottomLeft' : 'bottomRight';
  }
  const leftButton = jsxs(Button, {
    type: type,
    danger: danger,
    disabled: disabled,
    loading: loading,
    onClick: onClick,
    htmlType: htmlType,
    href: href,
    title: title,
    icon: children && leftButtonIcon ? leftButtonIcon : undefined,
    children: [leftButtonIcon && !children ? leftButtonIcon : undefined, children]
  });
  const rightButton = jsx(Button, {
    type: type,
    danger: danger,
    disabled: disabled,
    "aria-label": menuButtonLabel,
    children: icon ? icon : jsx(ChevronDownIcon, {})
  });
  const [leftButtonToRender, rightButtonToRender] = buttonsRender([leftButton, rightButton]);
  return jsxs(ButtonGroup, {
    ...restProps,
    className: classnames(prefixCls, className),
    children: [leftButtonToRender, overlay !== undefined ? jsx(Dropdown$1, {
      ...dropdownProps,
      overlay: overlay,
      children: rightButtonToRender
    }) : jsxs(Root$2, {
      ...dropdownMenuRootProps,
      children: [jsx(Trigger$1, {
        disabled: disabled,
        asChild: true,
        children: rightButtonToRender
      }), menu && /*#__PURE__*/React.cloneElement(menu, {
        align: menu.props.align || 'end'
      })]
    })]
  });
};

function _EMOTION_STRINGIFIED_CSS_ERROR__$5() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const BUTTON_HORIZONTAL_PADDING = 12;
function getSplitButtonEmotionStyles(classNamePrefix, theme) {
  const classDefault = `.${classNamePrefix}-btn`;
  const classPrimary = `.${classNamePrefix}-btn-primary`;
  const classDropdownTrigger = `.${classNamePrefix}-dropdown-trigger`;
  const classSmall = `.${classNamePrefix}-btn-group-sm`;
  const styles = {
    [classDefault]: {
      ...getDefaultStyles(theme),
      boxShadow: 'none',
      height: theme.general.heightSm,
      padding: `4px ${BUTTON_HORIZONTAL_PADDING}px`,
      '&:focus-visible': {
        outlineStyle: 'solid',
        outlineWidth: '2px',
        outlineOffset: '-2px',
        outlineColor: theme.colors.primary
      }
    },
    [classPrimary]: {
      ...getPrimaryStyles(theme),
      [`&:first-child`]: {
        borderRight: `1px solid ${theme.colors.actionPrimaryTextDefault}`,
        marginRight: 1
      },
      [classDropdownTrigger]: {
        borderLeft: `1px solid ${theme.colors.actionPrimaryTextDefault}`
      },
      '&:focus-visible': {
        outlineStyle: 'solid',
        outlineWidth: '1px',
        outlineOffset: '-3px',
        outlineColor: theme.colors.white
      }
    },
    [classDropdownTrigger]: {
      // Needs to be 1px less than our standard 8px to allow for the off-by-one border handling in this component.
      padding: 3,
      borderLeftColor: 'transparent',
      width: theme.general.heightSm
    },
    [`&${classSmall}`]: {
      [classDropdownTrigger]: {
        padding: 5
      }
    },
    '[disabled]': {
      ...getDisabledStyles(theme),
      [`&:first-child`]: {
        borderRight: `1px solid ${theme.colors.actionDisabledText}`,
        marginRight: 1
      },
      [classDropdownTrigger]: {
        borderLeft: `1px solid ${theme.colors.actionDisabledText}`
      }
    },
    [`${classDefault}:not(:first-child)`]: {
      width: theme.general.heightSm,
      padding: '3px !important'
    },
    ...getAnimationCss(theme.options.enableAnimation)
  };
  const importantStyles = importantify(styles);
  return /*#__PURE__*/css(importantStyles, process.env.NODE_ENV === "production" ? "" : ";label:getSplitButtonEmotionStyles;");
}
var _ref = process.env.NODE_ENV === "production" ? {
  name: "tp1ooh",
  styles: "display:inline-flex;position:relative;vertical-align:middle"
} : {
  name: "1kplxg4-SplitButton",
  styles: "display:inline-flex;position:relative;vertical-align:middle;label:SplitButton;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$5
};
const SplitButton = props => {
  const {
    theme,
    classNamePrefix
  } = useDesignSystemTheme();
  const {
    USE_NEW_ICONS: useNewIcons
  } = useDesignSystemFlags();
  const {
    children,
    icon,
    deprecatedMenu,
    type,
    loading,
    loadingButtonStyles,
    placement,
    dangerouslySetAntdProps,
    ...dropdownButtonProps
  } = props;

  // Size of button when loading only icon is shown
  const LOADING_BUTTON_SIZE = (useNewIcons ? theme.general.iconFontSizeNew : theme.general.iconFontSize) + 2 * BUTTON_HORIZONTAL_PADDING + 2 * theme.general.borderWidth;
  const [width, setWidth] = useState(LOADING_BUTTON_SIZE);

  // Set the width to the button's width in regular state to later use when in loading state
  // We do this to have just a loading icon in loading state at the normal width to avoid flicker and width changes in page
  const ref = useCallback(node => {
    if (node && !loading) {
      setWidth(node.getBoundingClientRect().width);
    }
  }, [loading]);
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx("div", {
      ref: ref,
      css: _ref,
      children: loading ? jsx(Button, {
        type: type === 'default' ? undefined : type,
        style: {
          width: width,
          ...(useNewIcons && {
            fontSize: theme.general.iconFontSizeNew
          }),
          ...loadingButtonStyles
        },
        loading: true,
        htmlType: props.htmlType,
        title: props.title,
        className: props.className,
        children: children
      }) : jsx(DropdownButton, {
        ...dropdownButtonProps,
        overlay: deprecatedMenu,
        trigger: ['click'],
        css: getSplitButtonEmotionStyles(classNamePrefix, theme),
        icon: jsx(ChevronDownIcon, {
          css: /*#__PURE__*/css({
            fontSize: theme.general.iconSize
          }, process.env.NODE_ENV === "production" ? "" : ";label:SplitButton;")
        }),
        placement: placement || 'bottomRight',
        type: type === 'default' ? undefined : type,
        leftButtonIcon: icon,
        ...dangerouslySetAntdProps,
        children: children
      })
    })
  });
};

const getSwitchWithLabelStyles = _ref => {
  let {
    clsPrefix,
    theme
  } = _ref;
  // Default value
  const SWITCH_WIDTH = 28;
  const styles = {
    display: 'flex',
    alignItems: 'center',
    // Switch is Off
    [`&.${clsPrefix}-switch`]: {
      backgroundColor: theme.colors.backgroundPrimary,
      border: `1px solid ${theme.colors.actionDefaultBorderDefault}`,
      [`.${clsPrefix}-switch-handle:before`]: {
        boxShadow: `0px 0px 0px 1px ${theme.colors.actionDefaultBorderDefault}`,
        transition: 'none'
      },
      [`&:hover:not(.${clsPrefix}-switch-disabled)`]: {
        backgroundColor: theme.colors.actionDefaultBackgroundHover,
        border: `1px solid ${theme.colors.actionPrimaryBackgroundHover}`,
        [`.${clsPrefix}-switch-handle:before`]: {
          boxShadow: `0px 0px 0px 1px ${theme.colors.actionPrimaryBackgroundHover}`
        }
      },
      [`&:active:not(.${clsPrefix}-switch-disabled)`]: {
        backgroundColor: theme.colors.actionDefaultBackgroundPress,
        border: `1px solid ${theme.colors.actionPrimaryBackgroundPress}`,
        [`.${clsPrefix}-switch-handle:before`]: {
          boxShadow: `0px 0px 0px 1px ${theme.colors.actionPrimaryBackgroundHover}`
        }
      },
      [`&:focus-visible`]: {
        border: `1px solid ${theme.colors.actionPrimaryBackgroundDefault}`,
        boxShadow: 'none',
        outlineStyle: 'solid',
        outlineWidth: '1px',
        outlineOffset: '1px',
        outlineColor: theme.colors.actionDefaultBorderFocus,
        [`.${clsPrefix}-switch-handle:before`]: {
          boxShadow: `0px 0px 0px 1px ${theme.colors.actionPrimaryBackgroundDefault}`
        }
      },
      [`&:focus`]: {
        boxShadow: 'none'
      }
    },
    // Switch is On
    [`&.${clsPrefix}-switch-checked`]: {
      backgroundColor: theme.colors.actionPrimaryBackgroundDefault,
      border: `1px solid ${theme.colors.actionPrimaryBackgroundDefault}`,
      [`&:hover:not(.${clsPrefix}-switch-disabled)`]: {
        backgroundColor: theme.colors.actionPrimaryBackgroundHover,
        border: `1px solid ${theme.colors.actionPrimaryBackgroundDefault}`
      },
      [`&:active:not(.${clsPrefix}-switch-disabled)`]: {
        backgroundColor: theme.colors.actionPrimaryBackgroundPress
      },
      [`.${clsPrefix}-switch-handle:before`]: {
        boxShadow: `0px 0px 0px 1px ${theme.colors.actionPrimaryBackgroundDefault}`
      },
      [`&.${clsPrefix}-switch-disabled`]: {
        backgroundColor: theme.colors.actionDisabledBackground,
        border: `1px solid ${theme.colors.actionDisabledBackground}`,
        [`.${clsPrefix}-switch-handle:before`]: {
          boxShadow: `0px 0px 0px 1px ${theme.colors.actionDisabledBackground}`
        }
      }
    },
    [`.${clsPrefix}-switch-handle:before`]: {
      backgroundColor: theme.colors.backgroundPrimary
    },
    [`&& + .${clsPrefix}-hint, && + .${clsPrefix}-form-message`]: {
      paddingLeft: theme.spacing.sm + SWITCH_WIDTH
    },
    [`&& + .${clsPrefix}-form-message`]: {
      marginTop: 0
    },
    [`.${clsPrefix}-click-animating-node`]: {
      animation: 'none'
    }
  };
  const importantStyles = importantify(styles);
  return /*#__PURE__*/css(importantStyles, process.env.NODE_ENV === "production" ? "" : ";label:getSwitchWithLabelStyles;");
};
const Switch = _ref2 => {
  let {
    dangerouslySetAntdProps,
    label,
    labelProps,
    ...props
  } = _ref2;
  const {
    theme,
    classNamePrefix
  } = useDesignSystemTheme();
  return label ? jsx(DesignSystemAntDConfigProvider, {
    children: jsxs("div", {
      css: getSwitchWithLabelStyles({
        clsPrefix: classNamePrefix,
        theme
      }),
      children: [jsx(Switch$1, {
        ...props,
        ...dangerouslySetAntdProps,
        css: /*#__PURE__*/css({
          ... /*#__PURE__*/css(getAnimationCss(theme.options.enableAnimation), process.env.NODE_ENV === "production" ? "" : ";label:Switch;"),
          ...getSwitchWithLabelStyles({
            clsPrefix: classNamePrefix,
            theme
          })
        }, process.env.NODE_ENV === "production" ? "" : ";label:Switch;")
      }), jsx(Label, {
        inline: true,
        ...labelProps,
        children: label
      })]
    })
  }) : jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Switch$1, {
      ...props,
      ...dangerouslySetAntdProps,
      css: /*#__PURE__*/css({
        ... /*#__PURE__*/css(getAnimationCss(theme.options.enableAnimation), process.env.NODE_ENV === "production" ? "" : ";label:Switch;"),
        ...getSwitchWithLabelStyles({
          clsPrefix: classNamePrefix,
          theme
        })
      }, process.env.NODE_ENV === "production" ? "" : ";label:Switch;")
    })
  });
};

function _EMOTION_STRINGIFIED_CSS_ERROR__$4() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }

// For performance, these styles are defined outside of the component so they are not redefined on every render.
// We're also using CSS Variables rather than any dynamic styles so that the style object remains static.
const tableStyles = {
  tableWrapper: process.env.NODE_ENV === "production" ? {
    name: "b4b8ls",
    styles: "&.table-isScrollable{overflow:auto;}display:flex;flex-direction:column;height:100%"
  } : {
    name: "1awn1ma-tableWrapper",
    styles: "&.table-isScrollable{overflow:auto;}display:flex;flex-direction:column;height:100%;label:tableWrapper;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$4
  },
  table: /*#__PURE__*/css({
    '.table-isScrollable &': {
      flex: 1,
      overflow: 'auto',
      // Adapted from: https://css-tricks.com/scroll-shadows-with-javascript/
      background: `
        // Top and bottom shadow masks
        linear-gradient(
          var(--table-background-color) 30%,
          transparent
        ),
        linear-gradient(
          transparent,
          var(--table-background-color) 70%
        ),

        // Left and right shadow masks
        linear-gradient(
          90deg,
          var(--table-background-color) 30%,
          transparent
        ),
        linear-gradient(
          90deg,
          transparent,
          var(--table-background-color) 70%
        ),

        // Top and bottom shadows
        radial-gradient(farthest-side at 50% 0%, var(--table-scroll-shadow-color) 0%, rgba(0, 0, 0, 0) 100%),
        radial-gradient(farthest-side at 50% 100%, var(--table-scroll-shadow-color) 0%, rgba(0, 0, 0, 0) 100%),
        
        // Left and right shadows
        radial-gradient(farthest-side at 0% 50%, var(--table-scroll-shadow-color) 0%, rgba(0, 0, 0, 0) 100%),
        radial-gradient(farthest-side at 100% 50%, var(--table-scroll-shadow-color) 0%, rgba(0, 0, 0, 0) 100%)
      `,
      backgroundRepeat: 'no-repeat',
      backgroundSize: `
        // Top and bottom shadow masks
        100% 20px,
        100% 20px,
        // Left and right shadow masks
        20px 100%,
        20px 100%,
        // Top and bottom shadows
        100% 10px,
        100% 10px,
        // Left and right shadows
        10px 100%,
        10px 100%
      `,
      backgroundAttachment: 'local, local, local, local, scroll, scroll, scroll, scroll'
    }
  }, process.env.NODE_ENV === "production" ? "" : ";label:table;"),
  row: process.env.NODE_ENV === "production" ? {
    name: "1aw6lkf",
    styles: "display:flex;&.table-isHeader{> *{background-color:var(--table-header-background-color);}.table-isScrollable &{position:sticky;top:0;z-index:1;}}.table-row-select-cell input[type=\"checkbox\"] ~ *{opacity:var(--row-checkbox-opacity, 0);}&:hover{&:not(.table-isHeader){background-color:var(--table-row-hover);}.table-row-select-cell input[type=\"checkbox\"] ~ *{opacity:1;}}.table-row-select-cell input[type=\"checkbox\"]:focus ~ *{opacity:1;}> *{padding-top:var(--table-row-vertical-padding);padding-bottom:var(--table-row-vertical-padding);border-bottom:1px solid;border-color:var(--table-separator-color);}"
  } : {
    name: "1vx4n96-row",
    styles: "display:flex;&.table-isHeader{> *{background-color:var(--table-header-background-color);}.table-isScrollable &{position:sticky;top:0;z-index:1;}}.table-row-select-cell input[type=\"checkbox\"] ~ *{opacity:var(--row-checkbox-opacity, 0);}&:hover{&:not(.table-isHeader){background-color:var(--table-row-hover);}.table-row-select-cell input[type=\"checkbox\"] ~ *{opacity:1;}}.table-row-select-cell input[type=\"checkbox\"]:focus ~ *{opacity:1;}> *{padding-top:var(--table-row-vertical-padding);padding-bottom:var(--table-row-vertical-padding);border-bottom:1px solid;border-color:var(--table-separator-color);};label:row;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$4
  },
  cell: process.env.NODE_ENV === "production" ? {
    name: "1n3v1dk",
    styles: "display:inline-grid;position:relative;flex:1;line-height:initial;box-sizing:border-box;padding-left:var(--table-spacing-sm);padding-right:var(--table-spacing-sm);word-break:break-word;overflow:hidden;& .anticon{vertical-align:text-bottom;}"
  } : {
    name: "q5g0tm-cell",
    styles: "display:inline-grid;position:relative;flex:1;line-height:initial;box-sizing:border-box;padding-left:var(--table-spacing-sm);padding-right:var(--table-spacing-sm);word-break:break-word;overflow:hidden;& .anticon{vertical-align:text-bottom;};label:cell;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$4
  },
  header: process.env.NODE_ENV === "production" ? {
    name: "ik7qgz",
    styles: "font-weight:bold;align-items:flex-end;display:flex;overflow:hidden;&[aria-sort]{cursor:pointer;user-select:none;}.table-header-text{color:var(--table-header-text-color);}.table-header-icon-container{color:var(--table-header-sort-icon-color);display:none;}&[aria-sort]:hover{.table-header-icon-container, .table-header-text{color:var(--table-header-focus-color);}}&[aria-sort]:active{.table-header-icon-container, .table-header-text{color:var(--table-header-active-color);}}&:hover, &[aria-sort=\"ascending\"], &[aria-sort=\"descending\"]{.table-header-icon-container{display:inline;}}"
  } : {
    name: "1xg6jn4-header",
    styles: "font-weight:bold;align-items:flex-end;display:flex;overflow:hidden;&[aria-sort]{cursor:pointer;user-select:none;}.table-header-text{color:var(--table-header-text-color);}.table-header-icon-container{color:var(--table-header-sort-icon-color);display:none;}&[aria-sort]:hover{.table-header-icon-container, .table-header-text{color:var(--table-header-focus-color);}}&[aria-sort]:active{.table-header-icon-container, .table-header-text{color:var(--table-header-active-color);}}&:hover, &[aria-sort=\"ascending\"], &[aria-sort=\"descending\"]{.table-header-icon-container{display:inline;}};label:header;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$4
  },
  headerButtonTarget: process.env.NODE_ENV === "production" ? {
    name: "sezlox",
    styles: "align-items:flex-end;display:flex;overflow:hidden;width:100%;justify-content:inherit;&:focus{.table-header-text{color:var(--table-header-focus-color);}.table-header-icon-container{color:var(--table-header-focus-color);display:inline;}}&:active{.table-header-icon-container, .table-header-text{color:var(--table-header-active-color);}}"
  } : {
    name: "1iv4trp-headerButtonTarget",
    styles: "align-items:flex-end;display:flex;overflow:hidden;width:100%;justify-content:inherit;&:focus{.table-header-text{color:var(--table-header-focus-color);}.table-header-icon-container{color:var(--table-header-focus-color);display:inline;}}&:active{.table-header-icon-container, .table-header-text{color:var(--table-header-active-color);}};label:headerButtonTarget;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$4
  },
  sortHeaderIcon: process.env.NODE_ENV === "production" ? {
    name: "1hdiaor",
    styles: "margin-left:var(--table-spacing-xs)"
  } : {
    name: "oruvmh-sortHeaderIcon",
    styles: "margin-left:var(--table-spacing-xs);label:sortHeaderIcon;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$4
  },
  checkboxCell: process.env.NODE_ENV === "production" ? {
    name: "4cdr0s",
    styles: "display:flex;align-items:center;flex:0;padding-left:var(--table-spacing-sm);padding-top:0;padding-bottom:0;min-width:var(--table-spacing-md);max-width:var(--table-spacing-md);box-sizing:content-box"
  } : {
    name: "17au6u2-checkboxCell",
    styles: "display:flex;align-items:center;flex:0;padding-left:var(--table-spacing-sm);padding-top:0;padding-bottom:0;min-width:var(--table-spacing-md);max-width:var(--table-spacing-md);box-sizing:content-box;label:checkboxCell;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$4
  },
  resizeHandleContainer: /*#__PURE__*/css({
    position: 'absolute',
    right: -3,
    top: 'var(--table-spacing-sm)',
    bottom: 'var(--table-spacing-sm)',
    width: 'var(--table-spacing-sm)',
    display: 'flex',
    justifyContent: 'center',
    cursor: 'col-resize',
    userSelect: 'none',
    touchAction: 'none',
    zIndex: 1
  }, process.env.NODE_ENV === "production" ? "" : ";label:resizeHandleContainer;"),
  resizeHandle: process.env.NODE_ENV === "production" ? {
    name: "55zery",
    styles: "width:1px;background:var(--table-resize-handle-color)"
  } : {
    name: "1ot7jju-resizeHandle",
    styles: "width:1px;background:var(--table-resize-handle-color);label:resizeHandle;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$4
  },
  paginationContainer: process.env.NODE_ENV === "production" ? {
    name: "ehlmid",
    styles: "display:flex;justify-content:flex-end;padding-top:var(--table-spacing-sm);padding-bottom:var(--table-spacing-sm)"
  } : {
    name: "p324df-paginationContainer",
    styles: "display:flex;justify-content:flex-end;padding-top:var(--table-spacing-sm);padding-bottom:var(--table-spacing-sm);label:paginationContainer;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$4
  }
};
var tableStyles$1 = tableStyles;

const TableContext = /*#__PURE__*/createContext({
  size: 'default'
});
const Table = /*#__PURE__*/forwardRef(function Table(_ref, ref) {
  let {
    children,
    size = 'default',
    someRowsSelected,
    style,
    pagination,
    empty,
    className,
    scrollable = false,
    headerHeight,
    ...rest
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  const flags = useDesignSystemFlags();
  return jsx(TableContext.Provider, {
    value: {
      size,
      someRowsSelected
    },
    children: jsxs("div", {
      ...rest,
      // This is a performance optimization; we want to statically create the styles for the table,
      // but for the dynamic theme values, we need to use CSS variables.
      // See: https://emotion.sh/docs/best-practices#advanced-css-variables-with-style
      style: {
        ...style,
        ['--table-background-color']: theme.colors.backgroundPrimary,
        ['--table-scroll-shadow-color']: theme.isDarkMode ? 'rgba(255, 255, 255, 0.07)' : 'rgba(31, 39, 45, 0.15)',
        ['--table-header-active-color']: theme.colors.actionDefaultTextPress,
        ['colorScheme']: theme.isDarkMode ? 'dark' : undefined,
        // This hex is pulled directly from the old table as a temporary style-matching measure.
        ['--table-header-background-color']: flags.USE_UPDATED_TABLE_STYLES || theme.isDarkMode ? theme.colors.backgroundPrimary : '#F2F5F7',
        ['--table-header-focus-color']: theme.colors.actionDefaultTextHover,
        ['--table-header-sort-icon-color']: theme.colors.textSecondary,
        ['--table-header-text-color']: theme.colors.actionDefaultTextDefault,
        ['--table-row-hover']: theme.colors.tableRowHover,
        ['--table-separator-color']: theme.colors.borderDecorative,
        ['--table-resize-handle-color']: flags.USE_UPDATED_TABLE_STYLES ? theme.colors.borderDecorative : theme.colors.grey400,
        ['--table-spacing-md']: `${theme.spacing.md}px`,
        ['--table-spacing-sm']: `${theme.spacing.sm}px`,
        ['--table-spacing-xs']: `${theme.spacing.xs}px`
      },
      css: tableStyles$1.tableWrapper,
      className: scrollable ? `table-isScrollable ${className}` : className,
      children: [jsxs("div", {
        role: "table",
        ref: ref,
        css: tableStyles$1.table,
        style:
        // TODO: The static pixel values here are for the top of the table, but this won't work
        // for headers with variable height (in those cases the shadow will be hidden behind the larger header)
        // We need to find a way to make this dynamic.
        headerHeight === 0 ? {
          background: theme.colors.backgroundPrimary
        } : size === 'small' ? {
          backgroundPosition: `
                    center ${headerHeight !== null && headerHeight !== void 0 ? headerHeight : 25}px,
                    center bottom,
                    left center,
                    right center,
                    center ${headerHeight !== null && headerHeight !== void 0 ? headerHeight : 25}px,
                    center bottom,
                    left center,
                    right center
                  `
        } : {
          backgroundPosition: `
                    center ${headerHeight !== null && headerHeight !== void 0 ? headerHeight : 36}px,
                    center bottom,
                    left center,
                    right center,
                    center ${headerHeight !== null && headerHeight !== void 0 ? headerHeight : 36}px,
                    center bottom,
                    left center,
                    right center
                  `
        },
        children: [children, empty && jsx("div", {
          css: /*#__PURE__*/css({
            padding: theme.spacing.lg
          }, process.env.NODE_ENV === "production" ? "" : ";label:Table;"),
          children: empty
        })]
      }), !empty && pagination && jsx("div", {
        css: tableStyles$1.paginationContainer,
        children: pagination
      })]
    })
  });
});

const TableCell = /*#__PURE__*/forwardRef(function (_ref, ref) {
  let {
    children,
    className,
    ellipsis = false,
    align = 'left',
    style,
    ...rest
  } = _ref;
  const {
    size
  } = useContext(TableContext);
  let typographySize = 'md';
  if (size === 'small') {
    typographySize = 'sm';
  }
  return jsx("div", {
    ...rest,
    css: [tableStyles$1.cell, process.env.NODE_ENV === "production" ? "" : ";label:TableCell;"],
    role: "cell",
    style: {
      textAlign: align,
      ...style
    },
    ref: ref,
    className: className,
    children: jsx(Typography.Text, {
      ellipsis: ellipsis,
      size: typographySize,
      title: ellipsis && typeof children === 'string' && children || undefined,
      children: children
    })
  });
});

function _EMOTION_STRINGIFIED_CSS_ERROR__$3() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
var _ref2 = process.env.NODE_ENV === "production" ? {
  name: "82a6rk",
  styles: "flex:1"
} : {
  name: "18ug1j7-component",
  styles: "flex:1;label:component;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$3
};
const TableFilterInput = /*#__PURE__*/forwardRef(function SearchInput(_ref, ref) {
  let {
    onSubmit,
    showSearchButton,
    className,
    onChange,
    onClear,
    ...inputProps
  } = _ref;
  const DEFAULT_WIDTH = 350;
  const handleChange = e => {
    // If the input is cleared, call the onClear handler, but only
    // if the event is not an input event -- which is the case when you click the
    // ant-provided (X) button.
    if (!e.target.value && e.nativeEvent instanceof InputEvent === false && onClear) {
      onClear === null || onClear === void 0 ? void 0 : onClear();
    } else {
      onChange === null || onChange === void 0 ? void 0 : onChange(e);
    }
  };
  let component = jsx(Input, {
    prefix: jsx(SearchIcon, {}),
    allowClear: true,
    ...inputProps,
    className: className,
    ref: ref,
    css: /*#__PURE__*/css({
      width: DEFAULT_WIDTH
    }, process.env.NODE_ENV === "production" ? "" : ";label:component;"),
    onChange: handleChange
  });
  if (showSearchButton) {
    component = jsxs(Input.Group, {
      style: {
        display: 'flex'
      },
      css: /*#__PURE__*/css({
        width: DEFAULT_WIDTH
      }, process.env.NODE_ENV === "production" ? "" : ";label:component;"),
      className: className,
      children: [jsx(Input, {
        allowClear: true,
        ...inputProps,
        ref: ref,
        css: _ref2,
        onChange: handleChange
      }), jsx(Button, {
        htmlType: "submit",
        children: jsx(SearchIcon, {})
      })]
    });
  }
  return jsx("form", {
    onSubmit: e => {
      e.preventDefault();
      onSubmit === null || onSubmit === void 0 ? void 0 : onSubmit();
    },
    children: component
  });
});

function _EMOTION_STRINGIFIED_CSS_ERROR__$2() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const tableFilterLayoutStyles = {
  layout: process.env.NODE_ENV === "production" ? {
    name: "1yb0qmd",
    styles: "display:flex;flex-direction:row;justify-content:space-between;margin-bottom:var(--table-filter-layout-group-margin)"
  } : {
    name: "bmua0k-layout",
    styles: "display:flex;flex-direction:row;justify-content:space-between;margin-bottom:var(--table-filter-layout-group-margin);label:layout;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$2
  },
  filters: process.env.NODE_ENV === "production" ? {
    name: "2pdmyz",
    styles: "display:flex;flex-wrap:wrap;flex-direction:row;align-items:center;gap:var(--table-filter-layout-item-gap);flex:1"
  } : {
    name: "i28ows-filters",
    styles: "display:flex;flex-wrap:wrap;flex-direction:row;align-items:center;gap:var(--table-filter-layout-item-gap);flex:1;label:filters;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$2
  },
  filterActions: process.env.NODE_ENV === "production" ? {
    name: "1ol8kzq",
    styles: "display:flex;gap:var(--table-filter-layout-item-gap);margin-left:var(--table-filter-layout-group-margin)"
  } : {
    name: "bcekwq-filterActions",
    styles: "display:flex;gap:var(--table-filter-layout-item-gap);margin-left:var(--table-filter-layout-group-margin);label:filterActions;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$2
  }
};
const TableFilterLayout = /*#__PURE__*/forwardRef(function TableFilterLayout(_ref, ref) {
  let {
    children,
    style,
    className,
    actions,
    ...rest
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  return jsxs("div", {
    ...rest,
    ref: ref,
    style: {
      ['--table-filter-layout-item-gap']: `${theme.spacing.sm}px`,
      ['--table-filter-layout-group-margin']: `${theme.spacing.md}px`,
      ...style
    },
    css: tableFilterLayoutStyles.layout,
    className: className,
    children: [jsx("div", {
      css: tableFilterLayoutStyles.filters,
      children: children
    }), actions && jsx("div", {
      css: tableFilterLayoutStyles.filterActions,
      children: actions
    })]
  });
});

const TableRowContext = /*#__PURE__*/createContext({
  isHeader: false
});
const TableRow = /*#__PURE__*/forwardRef(function TableRow(_ref, ref) {
  let {
    children,
    className,
    style,
    isHeader = false,
    verticalAlignment,
    ...rest
  } = _ref;
  const {
    size
  } = useContext(TableContext);
  const {
    theme
  } = useDesignSystemTheme();

  // Vertical only be larger if the row is a header AND size is large.
  const shouldUseLargeVerticalPadding = isHeader && size === 'default';
  let rowPadding;
  if (shouldUseLargeVerticalPadding) {
    rowPadding = theme.spacing.sm;
  } else if (size === 'default') {
    rowPadding = 6;
  } else {
    rowPadding = theme.spacing.xs;
  }
  return jsx(TableRowContext.Provider, {
    value: {
      isHeader
    },
    children: jsx("div", {
      ...rest,
      ref: ref,
      css: [tableStyles$1.row, process.env.NODE_ENV === "production" ? "" : ";label:TableRow;"],
      role: "row",
      style: {
        ...style,
        ['--table-row-vertical-padding']: `${rowPadding}px`
      },
      className: `${className} ${isHeader ? 'table-isHeader' : ''}`,
      children: children
    })
  });
});

const TableHeaderResizeHandle = /*#__PURE__*/forwardRef(function TableHeaderResizeHandle(_ref, ref) {
  let {
    style,
    resizeHandler,
    children,
    ...rest
  } = _ref;
  const {
    isHeader
  } = useContext(TableRowContext);
  if (!isHeader) {
    throw new Error('`TableHeaderResizeHandle` must be used within a `TableRow` with `isHeader` set to true.');
  }
  return jsx("div", {
    ...rest,
    ref: ref,
    onPointerDown: resizeHandler,
    css: tableStyles$1.resizeHandleContainer,
    style: style,
    role: "separator",
    children: jsx("div", {
      css: tableStyles$1.resizeHandle
    })
  });
});
const TableHeader = /*#__PURE__*/forwardRef(function TableHeader(_ref2, ref) {
  let {
    children,
    ellipsis = false,
    sortable,
    sortDirection,
    onToggleSort,
    style,
    className,
    resizable,
    resizeHandler,
    isResizing = false,
    align = 'left',
    ...rest
  } = _ref2;
  const {
    size
  } = useContext(TableContext);
  const {
    isHeader
  } = useContext(TableRowContext);
  if (!isHeader) {
    throw new Error('`TableHeader` must be used within a `TableRow` with `isHeader` set to true.');
  }
  let sortIcon = jsx(Fragment, {});
  // While most libaries use `asc` and `desc` for the sort value, the ARIA spec
  // uses `ascending` and `descending`.
  let ariaSort;
  if (sortable) {
    if (sortDirection === 'asc') {
      sortIcon = jsx(SortAscendingIcon, {});
      ariaSort = 'ascending';
    } else if (sortDirection === 'desc') {
      sortIcon = jsx(SortDescendingIcon, {});
      ariaSort = 'descending';
    } else if (sortDirection === 'none') {
      sortIcon = jsx(SortUnsortedIcon, {});
      ariaSort = 'none';
    }
  }
  let typographySize = 'md';
  if (size === 'small') {
    typographySize = 'sm';
  }
  const textContents = jsx(Typography.Text, {
    className: "table-header-text",
    ellipsis: ellipsis,
    size: typographySize,
    title: ellipsis && typeof children === 'string' && children || undefined,
    children: children
  });
  const resizeHandle = resizable ? jsx(TableHeaderResizeHandle, {
    resizeHandler: resizeHandler
  }) : null;
  return jsxs("div", {
    ...rest,
    ref: ref,
    css: [tableStyles$1.cell, tableStyles$1.header, process.env.NODE_ENV === "production" ? "" : ";label:TableHeader;"],
    role: "columnheader",
    "aria-sort": sortable && ariaSort || undefined,
    className: className,
    style: {
      justifyContent: align,
      textAlign: align,
      ...style
    },
    children: [sortable && !isResizing ? jsxs("div", {
      css: [tableStyles$1.headerButtonTarget, process.env.NODE_ENV === "production" ? "" : ";label:TableHeader;"],
      role: "button",
      tabIndex: 0,
      onClick: onToggleSort,
      onKeyDown: event => {
        if (sortable && (event.key === 'Enter' || event.key === ' ')) {
          event.preventDefault();
          return onToggleSort === null || onToggleSort === void 0 ? void 0 : onToggleSort(event);
        }
      },
      children: [textContents, jsx("span", {
        className: "table-header-icon-container",
        css: [tableStyles$1.sortHeaderIcon, process.env.NODE_ENV === "production" ? "" : ";label:TableHeader;"],
        children: sortIcon
      })]
    }) : textContents, resizeHandle]
  });
});

function _EMOTION_STRINGIFIED_CSS_ERROR__$1() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const TableRowActionStyles = {
  container: process.env.NODE_ENV === "production" ? {
    name: "137q2cp",
    styles: "width:32px;padding-top:0;padding-bottom:0;display:flex;align-items:center;justify-content:center"
  } : {
    name: "17o2n0c-container",
    styles: "width:32px;padding-top:0;padding-bottom:0;display:flex;align-items:center;justify-content:center;label:container;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$1
  }
};
const TableRowAction = /*#__PURE__*/forwardRef(function TableRowAction(_ref, ref) {
  let {
    children,
    style,
    className,
    ...rest
  } = _ref;
  return jsx("div", {
    ...rest,
    ref: ref,
    css: TableRowActionStyles.container,
    style: style,
    className: className,
    children: children
  });
});

/** @deprecated Use `TableRowAction` instead */
const TableRowMenuContainer = TableRowAction;

const TableRowSelectCell = /*#__PURE__*/forwardRef(function TableRowSelectCell(_ref, ref) {
  let {
    onChange,
    checked,
    indeterminate,
    noCheckbox,
    children,
    ...rest
  } = _ref;
  const {
    isHeader
  } = useContext(TableRowContext);
  const {
    someRowsSelected
  } = useContext(TableContext);
  if (typeof someRowsSelected === 'undefined') {
    throw new Error('`TableRowSelectCell` cannot be used unless `someRowsSelected` has been provided to the `Table` component, see documentation.');
  }
  if (!isHeader && indeterminate) {
    throw new Error('`TableRowSelectCell` cannot be used with `indeterminate` in a non-header row.');
  }
  return jsx("div", {
    ...rest,
    ref: ref,
    css: tableStyles$1.checkboxCell,
    style: {
      ['--row-checkbox-opacity']: someRowsSelected ? 1 : 0
    },
    role: isHeader ? 'columnheader' : 'cell'
    // TODO: Ideally we shouldn't need to specify this `className`, but it allows for row-hovering to reveal
    // the checkbox in `TableRow`'s CSS without extra JS pointerin/out events.
    ,
    className: "table-row-select-cell",
    children: !noCheckbox && jsx(Checkbox, {
      isChecked: checked || indeterminate && null,
      onChange: (_checked, event) => onChange === null || onChange === void 0 ? void 0 : onChange(event.nativeEvent)
    })
  });
});

function _EMOTION_STRINGIFIED_CSS_ERROR__() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const TableSkeletonStyles = {
  container: process.env.NODE_ENV === "production" ? {
    name: "6kz1wu",
    styles: "display:flex;flex-direction:column;align-items:flex-start"
  } : {
    name: "1we0er9-container",
    styles: "display:flex;flex-direction:column;align-items:flex-start;label:container;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__
  },
  cell: process.env.NODE_ENV === "production" ? {
    name: "1t820zr",
    styles: "width:100%;height:8px;border-radius:4px;background:var(--table-skeleton-color);margin-top:var(--table-skeleton-row-vertical-margin);margin-bottom:var(--table-skeleton-row-vertical-margin)"
  } : {
    name: "1m8dl5b-cell",
    styles: "width:100%;height:8px;border-radius:4px;background:var(--table-skeleton-color);margin-top:var(--table-skeleton-row-vertical-margin);margin-bottom:var(--table-skeleton-row-vertical-margin);label:cell;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__
  }
};
const TableSkeleton = _ref => {
  let {
    lines = 1,
    seed = '',
    style
    // TODO: Re-enable this after Clusters fixes tests: https://databricks.slack.com/archives/C04LYE3F8HX/p1679597678339659
    /** children, ...rest */
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  const {
    size
  } = useContext(TableContext);

  // This is a very simple PRNG that is seeded (so that the output is deterministic).
  // We need this in order to produce a random ragged edge for the table skeleton.
  function pseudoRandomNumberGeneratorFromSeed(seed) {
    // This is a simple way to get a consistent number from a string;
    // `charCodeAt` returns a number between 0 and 65535, and we then just add them all up.
    const seedValue = seed.split('').map(char => char.charCodeAt(0)).reduce((prev, curr) => prev + curr, 0);

    // This is a simple sine wave function that will always return a number between 0 and 1.
    // This produces a value akin to `Math.random()`, but has deterministic output.
    // Of course, sine curves are not a perfectly random distribution between 0 and 1, but
    // it's close enough for our purposes.
    return Math.sin(seedValue) / 2 + 0.5;
  }

  // This is a simple Fisher-Yates shuffler using the above PRNG.
  function shuffleArray(arr, seed) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(pseudoRandomNumberGeneratorFromSeed(seed + String(i)) * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }

  // Finally, we shuffle a list off offsets to apply to the widths of the cells.
  // This ensures that the cells are not all the same width, but that they are
  // random to produce a more realistic looking skeleton.
  const widths = shuffleArray([48, 24, 0], seed);
  return jsx("div", {
    // TODO: Re-enable this after Clusters fixes tests: https://databricks.slack.com/archives/C04LYE3F8HX/p1679597678339659
    // {...rest}
    css: TableSkeletonStyles.container,
    style: {
      ...style,
      // TODO: Pull this from the themes; it's not currently available.
      ['--table-skeleton-color']: theme.isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(31, 38, 45, 0.1)',
      ['--table-skeleton-row-vertical-margin']: size === 'small' ? '4px' : '6px'
    },
    children: [...Array(lines)].map((_, idx) => jsx("div", {
      css: [TableSkeletonStyles.cell, {
        width: `calc(100% - ${widths[idx % widths.length]}px)`
      }, process.env.NODE_ENV === "production" ? "" : ";label:TableSkeleton;"]
    }, idx))
  });
};

const getTabEmotionStyles = (clsPrefix, theme) => {
  const classTab = `.${clsPrefix}-tabs-tab`;
  const classButton = `.${clsPrefix}-tabs-tab-btn`;
  const classActive = `.${clsPrefix}-tabs-tab-active`;
  const classDisabled = `.${clsPrefix}-tabs-tab-disabled`;
  const classUnderline = `.${clsPrefix}-tabs-ink-bar`;
  const classClosable = `.${clsPrefix}-tabs-tab-with-remove`;
  const classNav = `.${clsPrefix}-tabs-nav`;
  const classCloseButton = `.${clsPrefix}-tabs-tab-remove`;
  const classAddButton = `.${clsPrefix}-tabs-nav-add`;
  const styles = {
    '&&': {
      overflow: 'unset'
    },
    [classTab]: {
      borderBottom: 'none',
      backgroundColor: 'transparent',
      border: 'none',
      paddingLeft: 0,
      paddingRight: 0,
      paddingTop: 6,
      paddingBottom: 6,
      marginRight: 24
    },
    [classButton]: {
      color: theme.colors.textSecondary,
      fontWeight: theme.typography.typographyBoldFontWeight,
      textShadow: 'none',
      fontSize: theme.typography.fontSizeMd,
      lineHeight: theme.typography.lineHeightBase,
      '&:hover': {
        color: theme.colors.actionDefaultTextHover
      },
      '&:active': {
        color: theme.colors.actionDefaultTextPress
      },
      outlineWidth: 2,
      outlineStyle: 'none',
      outlineColor: theme.colors.primary,
      outlineOffset: 2,
      '&:focus-visible': {
        outlineStyle: 'auto'
      }
    },
    [classActive]: {
      [classButton]: {
        color: theme.colors.textPrimary
      },
      // Use box-shadow instead of border to prevent it from affecting the size of the element, which results in visual
      // jumping when switching tabs.
      boxShadow: `inset 0 -3px 0 ${theme.colors.actionPrimaryBackgroundDefault}`
    },
    [classDisabled]: {
      [classButton]: {
        color: theme.colors.actionDisabledText,
        '&:hover': {
          color: theme.colors.actionDisabledText
        },
        '&:active': {
          color: theme.colors.actionDisabledText
        }
      }
    },
    [classUnderline]: {
      display: 'none'
    },
    [classClosable]: {
      borderTop: 'none',
      borderLeft: 'none',
      borderRight: 'none',
      background: 'none',
      paddingTop: 0,
      paddingBottom: 0,
      height: theme.general.heightSm
    },
    [classNav]: {
      height: theme.general.heightSm,
      '&::before': {
        borderColor: theme.colors.borderDecorative
      }
    },
    [classCloseButton]: {
      height: 24,
      width: 24,
      padding: 6,
      borderRadius: theme.borders.borderRadiusMd,
      marginTop: 0,
      marginRight: 0,
      marginBottom: 0,
      marginLeft: 4,
      '&:hover': {
        backgroundColor: theme.colors.actionDefaultBackgroundHover,
        color: theme.colors.actionDefaultTextHover
      },
      '&:active': {
        backgroundColor: theme.colors.actionDefaultBackgroundPress,
        color: theme.colors.actionDefaultTextPress
      },
      '&:focus-visible': {
        outlineWidth: 2,
        outlineStyle: 'solid',
        outlineColor: theme.colors.primary
      }
    },
    [classAddButton]: {
      backgroundColor: 'transparent',
      color: theme.colors.textValidationInfo,
      border: 'none',
      borderRadius: theme.borders.borderRadiusMd,
      margin: 4,
      height: 24,
      width: 24,
      padding: 0,
      minWidth: 'auto',
      '&:hover': {
        backgroundColor: theme.colors.actionDefaultBackgroundHover,
        color: theme.colors.actionDefaultTextHover
      },
      '&:active': {
        backgroundColor: theme.colors.actionDefaultBackgroundPress,
        color: theme.colors.actionDefaultTextPress
      },
      '&:focus-visible': {
        outlineWidth: 2,
        outlineStyle: 'solid',
        outlineColor: theme.colors.primary
      },
      '& > .anticon': {
        fontSize: 16
      }
    },
    ...getAnimationCss(theme.options.enableAnimation)
  };
  const importantStyles = importantify(styles);
  return importantStyles;
};
const TabPane = _ref => {
  let {
    children,
    ...props
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Tabs$1.TabPane, {
      closeIcon: jsx(CloseIcon, {
        css: /*#__PURE__*/css({
          fontSize: theme.general.iconSize
        }, process.env.NODE_ENV === "production" ? "" : ";label:TabPane;")
      })
      // Note: this component must accept the entire `props` object and spread it here, because Ant's Tabs components
      // injects extra props here (at the time of writing, `prefixCls`, `tabKey` and `id`).
      // However, we use a restricted TS interface to still discourage consumers of the library from passing in these props.
      ,
      ...props,
      ...props.dangerouslySetAntdProps,
      children: jsx(RestoreAntDDefaultClsPrefix, {
        children: children
      })
    })
  });
};
const Tabs = /* #__PURE__ */(() => {
  const Tabs = _ref2 => {
    let {
      editable = false,
      activeKey,
      defaultActiveKey,
      onChange,
      onEdit,
      children,
      destroyInactiveTabPane = false,
      dangerouslySetAntdProps = {},
      dangerouslyAppendEmotionCSS = {},
      ...props
    } = _ref2;
    const {
      theme,
      classNamePrefix
    } = useDesignSystemTheme();
    return jsx(DesignSystemAntDConfigProvider, {
      children: jsx(Tabs$1, {
        activeKey: activeKey,
        defaultActiveKey: defaultActiveKey,
        onChange: onChange,
        onEdit: onEdit,
        destroyInactiveTabPane: destroyInactiveTabPane,
        type: editable ? 'editable-card' : 'card',
        addIcon: jsx(PlusIcon, {
          css: /*#__PURE__*/css({
            fontSize: theme.general.iconSize
          }, process.env.NODE_ENV === "production" ? "" : ";label:Tabs;")
        }),
        css: [getTabEmotionStyles(classNamePrefix, theme), importantify(dangerouslyAppendEmotionCSS), process.env.NODE_ENV === "production" ? "" : ";label:Tabs;"],
        ...dangerouslySetAntdProps,
        ...props,
        children: children
      })
    });
  };
  Tabs.TabPane = TabPane;
  return Tabs;
})();

const colorMap = {
  default: 'tagDefault',
  brown: 'tagBrown',
  coral: 'tagCoral',
  charcoal: 'tagCharcoal',
  indigo: 'tagIndigo',
  lemon: 'tagLemon',
  lime: 'tagLime',
  pink: 'tagPink',
  purple: 'tagPurple',
  teal: 'tagTeal',
  turquoise: 'tagTurquoise'
};
const SIZE = 20;
function getTagEmotionStyles(theme) {
  let color = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : 'default';
  let clickable = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : false;
  let useNewIcons = arguments.length > 3 ? arguments[3] : undefined;
  let textColor = theme.colors.tagText;
  const backgroundColor = theme.colors[colorMap[color]];
  let iconHover = theme.colors.tagIconHover;
  let iconPress = theme.colors.tagIconPress;
  let tagHover = theme.colors.tagHover;
  let tagPress = theme.colors.tagPress;

  // Because the default tag background color changes depending on system theme, so do its other variables.
  if (color === 'default') {
    textColor = theme.colors.textPrimary;
    iconHover = theme.colors.actionTertiaryTextHover;
    iconPress = theme.colors.actionTertiaryTextPress;
  }

  // Because lemon is a light yellow, all its variables pull from the light mode palette, regardless of system theme.
  if (color === 'lemon') {
    textColor = lightColorList.textPrimary;
    iconHover = lightColorList.actionTertiaryTextHover;
    iconPress = lightColorList.actionTertiaryTextPress;
    tagHover = lightColorList.actionTertiaryBackgroundHover;
    tagPress = lightColorList.actionTertiaryBackgroundPress;
  }
  return {
    tag: {
      border: 'none',
      color: textColor,
      padding: '2px 4px',
      backgroundColor,
      borderRadius: theme.borders.borderRadiusMd,
      marginRight: 8,
      display: 'inline-block',
      cursor: clickable ? 'pointer' : 'default'
    },
    content: {
      display: 'flex',
      alignItems: 'center'
    },
    close: {
      height: useNewIcons ? theme.general.iconFontSizeNew : SIZE,
      width: useNewIcons ? theme.general.iconFontSizeNew : SIZE,
      lineHeight: useNewIcons ? `${theme.general.iconFontSizeNew}px` : theme.typography.lineHeightMd,
      padding: 0,
      color: textColor,
      fontSize: useNewIcons ? theme.general.iconFontSizeNew : SIZE,
      margin: '-2px -4px -2px 2px',
      borderTopRightRadius: theme.borders.borderRadiusMd,
      borderBottomRightRadius: theme.borders.borderRadiusMd,
      border: 'none',
      background: 'none',
      cursor: 'pointer',
      '&:hover': {
        backgroundColor: tagHover,
        color: iconHover
      },
      '&:active': {
        backgroundColor: tagPress,
        color: iconPress
      },
      '&:focus-visible': {
        outlineStyle: 'solid',
        outlineWidth: 1,
        outlineOffset: 1,
        outlineColor: theme.colors.actionDefaultBorderFocus
      },
      '.anticon': {
        verticalAlign: 0
      },
      ...(useNewIcons && {
        marginLeft: theme.spacing.xs,
        marginRight: -theme.spacing.xs
      })
    },
    text: {
      padding: 0,
      fontSize: theme.typography.fontSizeBase,
      lineHeight: theme.typography.lineHeightSm
    }
  };
}
function Tag(props) {
  const {
    theme
  } = useDesignSystemTheme();
  const {
    USE_NEW_ICONS: useNewIcons
  } = useDesignSystemFlags();
  const {
    color,
    children,
    closable,
    onClose,
    role = 'status',
    ...attributes
  } = props;
  const isClickable = Boolean(props.onClick);
  const styles = getTagEmotionStyles(theme, color, isClickable, useNewIcons);
  return jsx("div", {
    role: role,
    ...attributes,
    css: styles.tag,
    children: jsxs("div", {
      css: styles.content,
      children: [jsx("div", {
        css: styles.text,
        children: children
      }), closable && jsx("button", {
        css: styles.close,
        tabIndex: 0,
        onClick: e => {
          e.stopPropagation();
          if (onClose) {
            onClose();
          }
        },
        onMouseDown: e => {
          // Keeps dropdowns of any underlying select from opening.
          e.stopPropagation();
        },
        children: jsx(CloseIcon, {
          css: /*#__PURE__*/css({
            fontSize: useNewIcons ? theme.general.iconFontSizeNew - 4 : SIZE
          }, process.env.NODE_ENV === "production" ? "" : ";label:Tag;")
        })
      })]
    })
  });
}

const getStyles = theme => {
  return /*#__PURE__*/css({
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    whiteSpace: 'nowrap',
    border: `1px solid ${theme.colors.actionDefaultBorderDefault}`,
    borderRadius: theme.general.borderRadiusBase,
    backgroundColor: 'transparent',
    color: theme.colors.actionDefaultTextDefault,
    height: theme.general.heightSm,
    padding: '0 12px',
    fontSize: theme.typography.fontSizeBase,
    lineHeight: `${theme.typography.lineHeightBase}px`,
    '&[data-state="on"]': {
      backgroundColor: theme.colors.actionDefaultBackgroundPress,
      color: theme.colors.actionDefaultTextPress,
      borderColor: theme.colors.actionDefaultBorderPress
    },
    '&:hover': {
      cursor: 'pointer',
      color: theme.colors.actionDefaultTextHover,
      backgroundColor: theme.colors.actionDefaultBackgroundHover,
      borderColor: theme.colors.actionDefaultBorderHover,
      '& > svg': {
        stroke: theme.colors.actionDefaultBorderHover
      }
    },
    '&:disabled': {
      cursor: 'default',
      border: 'transparent',
      color: theme.colors.actionDisabledText,
      backgroundColor: theme.colors.actionDisabledBackground,
      '& > svg': {
        stroke: theme.colors.border
      }
    }
  }, process.env.NODE_ENV === "production" ? "" : ";label:getStyles;");
};
const RectangleSvg = props => jsx("svg", {
  width: "16",
  height: "16",
  viewBox: "0 0 16 16",
  fill: "none",
  xmlns: "http://www.w3.org/2000/svg",
  ...props,
  children: jsx("rect", {
    x: "0.5",
    y: "0.5",
    width: "15",
    height: "15",
    rx: "3.5"
  })
});
const ToggleButton = /*#__PURE__*/forwardRef((_ref, ref) => {
  let {
    children,
    pressed,
    defaultPressed,
    ...props
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  const [isPressed, setIsPressed] = React__default.useState(defaultPressed);
  useEffect(() => {
    setIsPressed(pressed);
  }, [pressed]);
  return jsxs(Toggle.Root, {
    css: getStyles(theme),
    ...props,
    pressed: isPressed,
    onPressedChange: pressed => {
      var _props$onPressedChang;
      (_props$onPressedChang = props.onPressedChange) === null || _props$onPressedChang === void 0 ? void 0 : _props$onPressedChang.call(props, pressed);
      setIsPressed(pressed);
    },
    ref: ref,
    children: [isPressed ? jsx(CheckIcon, {
      css: /*#__PURE__*/css({
        marginRight: theme.spacing.xs
      }, process.env.NODE_ENV === "production" ? "" : ";label:ToggleButton;")
    }) : jsx(RectangleSvg, {
      css: /*#__PURE__*/css({
        height: theme.typography.lineHeightBase,
        width: theme.typography.lineHeightSm,
        marginRight: theme.spacing.xs,
        stroke: theme.colors.border
      }, process.env.NODE_ENV === "production" ? "" : ";label:ToggleButton;")
    }), children]
  });
});

/**
 * These styles share some aspects with the styles in the main `Checkbox.tsx` component.
 * However, due to significant differences in the internal implementation and DOM structure of the Tree Checkbox and the
 * main Checkbox, we have forked the styles here.
 * Some notable differences are:
 * 1. Tree checkbox does not have a wrapper div
 * 2. Tree checkbox does not use a hidden input element
 * 3. Tree checkbox does not support the disabled state.
 * 4. Tree checkbox does not support keyboard focus
 */
function getTreeCheckboxEmotionStyles(clsPrefix, theme) {
  const classRoot = `.${clsPrefix}`;
  const classInner = `.${clsPrefix}-inner`;
  const classIndeterminate = `.${clsPrefix}-indeterminate`;
  const classChecked = `.${clsPrefix}-checked`;
  const classDisabled = `.${clsPrefix}-disabled`;
  const styles = {
    [`${classRoot} > ${classInner}`]: {
      backgroundColor: theme.colors.actionDefaultBackgroundDefault,
      borderColor: theme.colors.actionDefaultBorderDefault
    },
    // Hover
    [`${classRoot}:hover > ${classInner}`]: {
      backgroundColor: theme.colors.actionDefaultBackgroundHover,
      borderColor: theme.colors.actionDefaultBorderHover
    },
    // Mouse pressed
    [`${classRoot}:active > ${classInner}`]: {
      backgroundColor: theme.colors.actionDefaultBackgroundPress,
      borderColor: theme.colors.actionDefaultBorderPress
    },
    // Checked state
    [`${classChecked} > ${classInner}`]: {
      backgroundColor: theme.colors.actionPrimaryBackgroundDefault,
      borderColor: 'transparent'
    },
    // Checked hover
    [`${classChecked}:hover > ${classInner}`]: {
      backgroundColor: theme.colors.actionPrimaryBackgroundHover,
      borderColor: 'transparent'
    },
    // Checked and mouse pressed
    [`${classChecked}:active > ${classInner}`]: {
      backgroundColor: theme.colors.actionPrimaryBackgroundPress,
      borderColor: 'transparent'
    },
    // Indeterminate
    [`${classIndeterminate} > ${classInner}`]: {
      backgroundColor: theme.colors.primary,
      borderColor: theme.colors.primary,
      // The after pseudo-element is used for the check image itself
      '&:after': {
        backgroundColor: theme.colors.white,
        height: '3px',
        width: '8px',
        borderRadius: '4px'
      }
    },
    // Indeterminate hover
    [`${classIndeterminate}:hover > ${classInner}`]: {
      backgroundColor: theme.colors.actionPrimaryBackgroundHover,
      borderColor: 'transparent'
    },
    // Indeterminate and mouse pressed
    [`${classIndeterminate}:active > ${classInner}`]: {
      backgroundColor: theme.colors.actionPrimaryBackgroundPress
    },
    // Disabled
    [`${classDisabled} > ${classInner}, ${classDisabled}:hover > ${classInner}, ${classDisabled}:active > ${classInner}`]: {
      backgroundColor: theme.colors.actionDisabledBackground
    },
    ...getAnimationCss(theme.options.enableAnimation)
  };
  return styles;
}
function getTreeEmotionStyles(clsPrefix, theme, size, useNewTree) {
  const classNode = `.${clsPrefix}-tree-treenode`;
  const classNodeSelected = `.${clsPrefix}-tree-treenode-selected`;
  const classNodeActive = `.${clsPrefix}-tree-treenode-active`;
  const classNodeDisabled = `.${clsPrefix}-tree-treenode-disabled`;
  const classContent = `.${clsPrefix}-tree-node-content-wrapper`;
  const classSelected = `.${clsPrefix}-tree-node-selected`;
  const classSwitcher = `.${clsPrefix}-tree-switcher`;
  const classSwitcherNoop = `.${clsPrefix}-tree-switcher-noop`;
  const classFocused = `.${clsPrefix}-tree-focused`;
  const classCheckbox = `.${clsPrefix}-tree-checkbox`;
  const classUnselectable = `.${clsPrefix}-tree-unselectable`;
  const classIndent = `.${clsPrefix}-tree-indent-unit`;
  const NODE_SIZE = size === 'small' ? 24 : 32;
  const ICON_FONT_SIZE = useNewTree ? 16 : 24;
  const BORDER_WIDTH = 4;
  const styles = {
    // Basic node
    [classNode]: {
      height: NODE_SIZE,
      width: '100%',
      padding: 0,
      paddingLeft: BORDER_WIDTH,
      display: 'flex',
      alignItems: 'center',
      // Ant tree renders some hidden tree nodes (presumably for internal purposes). Setting these to width: 100% causes
      // overflow, so we need to reset here.
      '&[aria-hidden=true]': {
        width: 'auto'
      },
      '&:hover': {
        backgroundColor: theme.colors.actionTertiaryBackgroundHover
      },
      '&:active': {
        backgroundColor: theme.colors.actionTertiaryBackgroundPress
      }
    },
    [`&${classUnselectable}`]: {
      // Remove hover and press styles if tree nodes are not selectable
      [classNode]: {
        '&:hover': {
          backgroundColor: 'transparent'
        },
        '&:active': {
          backgroundColor: 'transparent'
        }
      },
      [classContent]: {
        cursor: 'default'
      },
      // Unselectable nodes don't have any background, so the switcher looks better with rounded corners.
      [classSwitcher]: {
        borderRadius: theme.borders.borderRadiusMd
      }
    },
    // The "active" node is the one that is currently focused via keyboard navigation. We give it the same visual
    // treatment as the mouse hover style.
    [classNodeActive]: {
      backgroundColor: theme.colors.actionTertiaryBackgroundHover
    },
    // The "selected" node is one that has either been clicked on, or selected via pressing enter on the keyboard.
    [classNodeSelected]: {
      backgroundColor: theme.colors.actionTertiaryBackgroundPress,
      borderLeft: `${BORDER_WIDTH}px solid ${theme.colors.actionPrimaryBackgroundDefault}`,
      paddingLeft: 0,
      // When hovering over a selected node, we still want it to look selected
      '&:hover': {
        backgroundColor: theme.colors.actionTertiaryBackgroundPress
      }
    },
    [classSelected]: {
      background: 'none'
    },
    [classNodeDisabled]: {
      '&:hover': {
        backgroundColor: 'transparent'
      },
      '&:active': {
        backgroundColor: 'transparent'
      }
    },
    [classContent]: {
      lineHeight: `${NODE_SIZE}px`,
      ...(!useNewTree && {
        padding: 0
      }),
      // The content label is the interactive element, so we want it to fill the node to maximise the click area.
      width: '100%',
      '&:hover': {
        backgroundColor: 'transparent'
      },
      '&:active': {
        backgroundColor: 'transparent'
      }
    },
    // TODO(FEINF-1595): Temporary style for now
    ...(useNewTree && {
      [`${classSwitcherNoop} + ${classContent}, ${classSwitcherNoop} + ${classCheckbox}`]: {
        marginLeft: NODE_SIZE + 4
      }
    }),
    [classSwitcher]: {
      height: NODE_SIZE,
      width: NODE_SIZE,
      paddingTop: (NODE_SIZE - ICON_FONT_SIZE) / 2,
      marginRight: theme.spacing.xs,
      color: theme.colors.textSecondary,
      backgroundColor: 'transparent',
      // Keyboard navigation only allows moving between entire nodes, not between the switcher and label directly.
      // However, under mouse control, the two can still be clicked separately. We apply hover and press treatment
      // here to indicate to mouse users that the switcher is clickable.
      '&:hover': {
        backgroundColor: theme.colors.actionTertiaryBackgroundHover
      },
      '&:active': {
        backgroundColor: theme.colors.actionTertiaryBackgroundPress
      }
    },
    [classSwitcherNoop]: {
      ...(useNewTree && {
        display: 'none'
      }),
      '&:hover': {
        backgroundColor: 'transparent'
      },
      '&:active': {
        backgroundColor: 'transparent'
      }
    },
    [`&${classFocused}`]: {
      backgroundColor: 'transparent',
      outlineWidth: 2,
      outlineOffset: 1,
      outlineColor: theme.colors.primary,
      outlineStyle: 'solid'
    },
    [classCheckbox]: {
      marginTop: 0,
      marginBottom: 0,
      marginRight: theme.spacing.sm
    },
    // Vertical line
    ...(useNewTree && {
      [classIndent]: {
        width: size === 'small' ? 24 : 28
      },
      [`${classIndent}:before`]: {
        height: '100%'
      }
    }),
    ...getTreeCheckboxEmotionStyles(`${clsPrefix}-tree-checkbox`, theme),
    ...getAnimationCss(theme.options.enableAnimation)
  };
  const importantStyles = importantify(styles);
  return /*#__PURE__*/css(importantStyles, process.env.NODE_ENV === "production" ? "" : ";label:getTreeEmotionStyles;");
}
const SHOW_LINE_DEFAULT = {
  showLeafIcon: false
};

// @ts-expect-error: Tree doesn't expose a proper type
const Tree = /*#__PURE__*/forwardRef(function Tree(_ref, ref) {
  let {
    treeData,
    defaultExpandedKeys,
    defaultSelectedKeys,
    defaultCheckedKeys,
    disabled = false,
    mode = 'default',
    size = 'default',
    showLine,
    dangerouslySetAntdProps,
    ...props
  } = _ref;
  const {
    theme,
    classNamePrefix
  } = useDesignSystemTheme();
  const {
    USE_NEW_TREE
  } = useDesignSystemFlags();
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Tree$1, {
      treeData: treeData,
      defaultExpandedKeys: defaultExpandedKeys,
      defaultSelectedKeys: defaultSelectedKeys,
      defaultCheckedKeys: defaultCheckedKeys,
      disabled: disabled,
      css: getTreeEmotionStyles(classNamePrefix, theme, size, USE_NEW_TREE),
      switcherIcon: jsx(ChevronDownIcon, {
        css: /*#__PURE__*/css({
          fontSize: `${USE_NEW_TREE ? 16 : 24}px!important`
        }, process.env.NODE_ENV === "production" ? "" : ";label:Tree;")
      }),
      tabIndex: 0,
      selectable: mode === 'selectable' || mode === 'multiselectable',
      checkable: mode === 'checkable',
      multiple: mode === 'multiselectable'
      // With the library flag, defaults to showLine = true. The status quo default is showLine = false.
      ,
      showLine: USE_NEW_TREE ? showLine !== null && showLine !== void 0 ? showLine : SHOW_LINE_DEFAULT : showLine !== null && showLine !== void 0 ? showLine : false,
      ...dangerouslySetAntdProps,
      ...props,
      ref: ref
    })
  });
});

export { Accordion, AccordionPanel, Alert, AlignCenterIcon, AlignLeftIcon, AlignRightIcon, AppIcon, ApplyDesignSystemContextOverrides, ApplyDesignSystemFlags, ArrowDownIcon, ArrowLeftIcon, ArrowRightIcon, ArrowUpIcon, ArrowsUpDownIcon, AutoComplete, BarChartIcon, BeakerIcon, BinaryIcon, BoldIcon, BookIcon, BookmarkFillIcon, BookmarkIcon, BracketsCurlyIcon, BracketsSquareIcon, BracketsXIcon, Breadcrumb, BriefcaseFillIcon, BriefcaseIcon, Button, CalendarEventIcon, CalendarIcon, CaretDownSquareIcon, CaretUpSquareIcon, CatalogIcon, ChartLineIcon, CheckCircleBadgeIcon, CheckCircleFillIcon, CheckCircleIcon, CheckIcon, CheckLineIcon, Checkbox, ChecklistIcon, ChevronDoubleDownIcon, ChevronDoubleLeftIcon, ChevronDoubleRightIcon, ChevronDoubleUpIcon, ChevronDownIcon, ChevronLeftIcon, ChevronRightIcon, ChevronUpIcon, CircleIcon, ClipboardIcon, ClockIcon, CloseIcon, CloudDownloadIcon, CloudIcon, CloudKeyIcon, CloudModelIcon, CloudOffIcon, CloudUploadIcon, CodeIcon, Col, ColorFillIcon, ConnectIcon, Content, CopyIcon, CursorIcon, CursorPagination, DIcon, DU_BOIS_ENABLE_ANIMATION_CLASSNAME, DagIcon, DangerFillIcon, DangerIcon, DangerModal, DashIcon, DashboardIcon, DataIcon, DatabaseIcon, DecimalIcon, DesignSystemAntDConfigProvider, DesignSystemContext, DesignSystemProvider, DesignSystemThemeContext, DesignSystemThemeProvider, DialogCombobox, DialogComboboxButtonContainer, DialogComboboxContent, DialogComboboxCountBadge, DialogComboboxLoadingSpinner, DialogComboboxOptionControlledList, DialogComboboxOptionList, DialogComboboxOptionListCheckboxItem, DialogComboboxOptionListSearch, DialogComboboxOptionListSelectItem, DialogComboboxSectionHeader, DialogComboboxSeparator, DialogComboboxTrigger, DotsCircleIcon, DownloadIcon, DragIcon, Drawer, Dropdown, DropdownMenu, DuboisDatePicker, Empty, ExpandLessIcon, ExpandMoreIcon, FileCodeIcon, FileDocumentIcon, FileIcon, FileImageIcon, FileModelIcon, FilterIcon, FolderFillIcon, FolderIcon, FontIcon, ForkIcon, Form, FormDubois, FormUI, FullscreenExitIcon, FullscreenIcon, FunctionIcon, GearFillIcon, GearIcon, GiftIcon, GridDashIcon, GridIcon, H1Icon, H2Icon, H3Icon, Header$1 as Header, HistoryIcon, HomeIcon, Icon, ImageIcon, IndentDecreaseIcon, IndentIncreaseIcon, InfinityIcon, InfoFillIcon, InfoIcon, Input, ItalicIcon, KeyboardIcon, LayerIcon, Layout, LegacyDatePicker, LegacyTable, LettersIcon, LightningIcon, LinkIcon, LinkOffIcon, ListBorderIcon, ListIcon, LoadingIcon, LockFillIcon, LockIcon, LockUnlockedIcon, MIcon, Menu, MenuIcon, MinusBoxIcon, MinusCircleFillIcon, MinusCircleIcon, Modal, ModelsIcon, Nav, NavButton, NewWindowIcon, NoIcon, NotebookIcon, NotificationIcon, NotificationOffIcon, NotificationV2, NumbersIcon, OfficeIcon, OptGroup, Option, OverflowIcon, PageBottomIcon, PageFirstIcon, PageLastIcon, PageTopIcon, PageWrapper, Pagination, Panel, PanelBody, PanelHeader, PanelHeaderButtons, PanelHeaderTitle, PencilIcon, PinCancelIcon, PinFillIcon, PinIcon, PipelineIcon, PlayCircleFillIcon, PlayCircleIcon, PlayIcon, PlugIcon, PlusCircleFillIcon, PlusCircleIcon, PlusIcon, PlusSquareIcon, Popover, PopoverV2, QueryEditorIcon, QueryIcon, QuestionMarkFillIcon, QuestionMarkIcon, RHFControlledComponents, ROW_GUTTER_SIZE, Radio, ReaderModeIcon, RedoIcon, RefreshIcon, ReposIcon, RestoreAntDDefaultClsPrefix, Row, SaveIcon, SchoolIcon, SearchIcon, SecurityIcon, SegmentedControlButton, SegmentedControlGroup, Select, SelectOptGroup, SelectOption, ShareIcon, Sidebar, SidebarAutoIcon, SidebarCollapseIcon, SidebarExpandIcon, SidebarIcon, Skeleton, SlidersIcon, SortAscendingIcon, SortDescendingIcon, SortUnsortedIcon, Space, Spacer, SpeechBubbleIcon, SpeechBubblePlusIcon, Spinner, SplitButton, StarFillIcon, StarIcon, StopCircleFillIcon, StopCircleIcon, StopIcon, StorefrontIcon, StreamIcon, Switch, SyncIcon, TabPane, Table, TableCell, TableContext, TableFilterInput, TableFilterLayout, TableHeader, TableIcon, TableRow, TableRowAction, TableRowContext, TableRowMenuContainer, TableRowSelectCell, TableSkeleton, Tabs, Tag, TextBoxIcon, ToggleButton, Tooltip, TrashIcon, Tree, TreeIcon, Typography, UnderlineIcon, UndoIcon, UploadIcon, UsbIcon, UserBadgeIcon, UserCircleIcon, UserGroupIcon, UserIcon, VariableIcon, VisibleIcon, VisibleOffIcon, WarningFillIcon, WarningIcon, WorkspacesIcon, XCircleFillIcon, XCircleIcon, ZoomInIcon, ZoomOutIcon, __INTERNAL_DO_NOT_USE_DEDUPE__Group, __INTERNAL_DO_NOT_USE__FormItem, __INTERNAL_DO_NOT_USE__Group, __INTERNAL_DO_NOT_USE__HorizontalGroup, __INTERNAL_DO_NOT_USE__Password, __INTERNAL_DO_NOT_USE__TextArea, __INTERNAL_DO_NOT_USE__VerticalGroup, getAnimationCss, getButtonEmotionStyles, getPaginationEmotionStyles, getRadioStyles, getTabEmotionStyles, getWrapperStyle, useAntDConfigProviderContext, useDesignSystemFlags, useNotification, useThemedStyles, visuallyHidden, withNotifications };
//# sourceMappingURL=index.js.map
