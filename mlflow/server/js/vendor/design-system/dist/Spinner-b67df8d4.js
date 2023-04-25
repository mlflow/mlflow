import React__default, { createContext, useMemo, useEffect, useContext, forwardRef } from 'react';
import { jsx, jsxs } from '@emotion/react/jsx-runtime';
import { useTheme, ThemeProvider, css, keyframes } from '@emotion/react';
import { notification, ConfigProvider, Button as Button$1, Checkbox as Checkbox$1, Typography as Typography$1, Tooltip as Tooltip$1, Input as Input$1 } from 'antd';
import classnames from 'classnames';
import _isNil from 'lodash/isNil';
import _endsWith from 'lodash/endsWith';
import _isBoolean from 'lodash/isBoolean';
import _isNumber from 'lodash/isNumber';
import _isString from 'lodash/isString';
import _mapValues from 'lodash/mapValues';
import unitless from '@emotion/unitless';
import chroma from 'chroma-js';
import AntDIcon from '@ant-design/icons';

// Border variables
const borders = {
  borderRadiusMd: 4
};

// eslint-disable-next-line import/no-anonymous-default-export
var borders$1 = {
  ...borders
};

// Exported from go/designsystem/colorsheet
const colorPalettePrimary = {
  primary: '#2272B4',
  blue100: '#F0F8FF',
  blue200: '#D7EDFE',
  blue300: '#BAE1FC',
  blue400: '#8ACAFF',
  blue500: '#4299E0',
  blue600: '#2272B4',
  blue700: '#0E538B',
  blue800: '#04355D',
  green100: '#F3FCF6',
  green200: '#D4F7DF',
  green300: '#B1ECC5',
  green400: '#8DDDA8',
  green500: '#3CAA60',
  green600: '#277C43',
  green700: '#115026',
  green800: '#093919',
  white: '#FFFFFF',
  grey100: '#F2F5F7',
  grey200: '#E4ECF1',
  grey300: '#CDDAE5',
  grey400: '#BDCDDB',
  grey500: '#8196A7',
  grey600: '#5D7283',
  grey700: '#44535F',
  grey800: '#1F272D',
  red100: '#FFF5F7',
  red200: '#FDE2E8',
  red300: '#FBD0D8',
  red400: '#F792A6',
  red500: '#E65B77',
  red600: '#C82D4C',
  red700: '#9E102C',
  red800: '#630316',
  yellow100: '#FFF9EB',
  yellow200: '#FCEACA',
  yellow300: '#F8D4A5',
  yellow400: '#F2BE88',
  yellow500: '#DE7921',
  yellow600: '#BE501E',
  yellow700: '#93320B',
  yellow800: '#5F1B02'
};
const colorPaletteSecondary = {
  brown: '#A6630C',
  coral: '#C83243',
  charcoal: '#5D7283',
  indigo: '#434A93',
  lemon: '#FACB66',
  lime: '#308613',
  pink: '#B45091',
  purple: '#8A63BF',
  teal: '#04867D',
  turquoise: '#157CBC'
};
const lightColorList = {
  backgroundPrimary: colorPalettePrimary.white,
  actionDangerPrimaryBackgroundDefault: colorPalettePrimary.red600,
  actionDangerPrimaryBackgroundHover: colorPalettePrimary.red700,
  actionDangerPrimaryBackgroundPress: colorPalettePrimary.red800,
  actionDangerDefaultBackgroundDefault: chroma(colorPalettePrimary.red600).alpha(0).hex(),
  actionDangerDefaultBackgroundHover: chroma(colorPalettePrimary.red600).alpha(0.08).hex(),
  actionDangerDefaultBackgroundPress: chroma(colorPalettePrimary.red600).alpha(0.16).hex(),
  actionDangerDefaultBorderDefault: colorPalettePrimary.red600,
  actionDangerDefaultBorderHover: colorPalettePrimary.red700,
  actionDangerDefaultBorderPress: colorPalettePrimary.red800,
  actionDangerDefaultTextDefault: colorPalettePrimary.red600,
  actionDangerDefaultTextHover: colorPalettePrimary.red700,
  actionDangerDefaultTextPress: colorPalettePrimary.red800,
  actionDefaultBackgroundDefault: chroma(colorPalettePrimary.blue800).alpha(0).hex(),
  actionDefaultBackgroundHover: chroma(colorPalettePrimary.blue600).alpha(0.08).hex(),
  actionDefaultBackgroundPress: chroma(colorPalettePrimary.blue600).alpha(0.16).hex(),
  actionDefaultBorderDefault: colorPalettePrimary.grey300,
  actionDefaultBorderFocus: colorPalettePrimary.blue600,
  actionDefaultBorderHover: colorPalettePrimary.blue700,
  actionDefaultBorderPress: colorPalettePrimary.blue800,
  actionDefaultTextDefault: colorPalettePrimary.grey800,
  actionDefaultTextHover: colorPalettePrimary.blue700,
  actionDefaultTextPress: colorPalettePrimary.blue800,
  actionDisabledBackground: colorPalettePrimary.grey200,
  actionDisabledText: colorPalettePrimary.grey400,
  actionPrimaryBackgroundDefault: colorPalettePrimary.blue600,
  actionPrimaryBackgroundHover: colorPalettePrimary.blue700,
  actionPrimaryBackgroundPress: colorPalettePrimary.blue800,
  actionPrimaryTextDefault: colorPalettePrimary.white,
  actionPrimaryTextHover: colorPalettePrimary.white,
  actionPrimaryTextPress: colorPalettePrimary.white,
  actionTertiaryBackgroundDefault: chroma(colorPalettePrimary.blue600).alpha(0).hex(),
  actionTertiaryBackgroundHover: chroma(colorPalettePrimary.blue600).alpha(0.08).hex(),
  actionTertiaryBackgroundPress: chroma(colorPalettePrimary.blue600).alpha(0.16).hex(),
  actionTertiaryTextDefault: colorPalettePrimary.blue600,
  actionTertiaryTextHover: colorPalettePrimary.blue700,
  actionTertiaryTextPress: colorPalettePrimary.blue800,
  backgroundDanger: chroma(colorPalettePrimary.red100).alpha(0.08).hex(),
  backgroundSecondary: colorPalettePrimary.grey100,
  backgroundWarning: chroma(colorPalettePrimary.yellow100).alpha(0.08).hex(),
  backgroundValidationDanger: colorPalettePrimary.red100,
  backgroundValidationSuccess: colorPalettePrimary.blue100,
  backgroundValidationWarning: colorPalettePrimary.yellow100,
  border: colorPalettePrimary.grey300,
  borderDecorative: colorPalettePrimary.grey300,
  borderValidationDanger: colorPalettePrimary.red300,
  borderValidationWarning: colorPalettePrimary.yellow300,
  textPrimary: colorPalettePrimary.grey800,
  textSecondary: colorPalettePrimary.grey600,
  textPlaceholder: colorPalettePrimary.grey400,
  textValidationDanger: colorPalettePrimary.red600,
  textValidationSuccess: colorPalettePrimary.green600,
  textValidationWarning: colorPalettePrimary.yellow600,
  textValidationInfo: colorPalettePrimary.grey600,
  overlayOverlay: chroma(colorPalettePrimary.grey100).alpha(0.72).hex(),
  // Tags
  tagDefault: chroma(colorPalettePrimary.grey600).alpha(0.08).hex(),
  tagBrown: colorPaletteSecondary.brown,
  tagCoral: colorPaletteSecondary.coral,
  tagCharcoal: colorPalettePrimary.grey600,
  tagIndigo: colorPaletteSecondary.indigo,
  tagLemon: colorPaletteSecondary.lemon,
  tagLime: colorPaletteSecondary.lime,
  tagPink: colorPaletteSecondary.pink,
  tagPurple: colorPaletteSecondary.purple,
  tagTeal: colorPaletteSecondary.teal,
  tagTurquoise: colorPaletteSecondary.turquoise,
  tagText: colorPalettePrimary.white,
  tagHover: chroma(colorPalettePrimary.blue400).alpha(0.08).hex(),
  tagPress: chroma(colorPalettePrimary.blue400).alpha(0.16).hex(),
  tagIconHover: chroma(colorPalettePrimary.white).alpha(0.8).hex(),
  tagIconPress: chroma(colorPalettePrimary.white).alpha(0.76).hex(),
  // Typography
  typographyCodeBg: chroma(colorPalettePrimary.grey600).alpha(0.08).hex(),
  // Table
  tableSeparatorColor: colorPalettePrimary.grey200,
  tableRowHover: chroma(colorPalettePrimary.grey600).alpha(0.04).hex(),
  tooltipBackgroundTooltip: colorPalettePrimary.grey800
};
const darkColorList = {
  actionDangerPrimaryBackgroundDefault: chroma(colorPalettePrimary.red400).alpha(0.84).hex(),
  actionDangerPrimaryBackgroundHover: chroma(colorPalettePrimary.red400).alpha(0.72).hex(),
  actionDangerPrimaryBackgroundPress: chroma(colorPalettePrimary.red400).alpha(0.6).hex(),
  actionDangerDefaultBackgroundDefault: chroma(colorPalettePrimary.red400).alpha(0).hex(),
  actionDangerDefaultBackgroundHover: chroma(colorPalettePrimary.red400).alpha(0.08).hex(),
  actionDangerDefaultBackgroundPress: chroma(colorPalettePrimary.red400).alpha(0.16).hex(),
  actionDangerDefaultBorderDefault: colorPalettePrimary.red400,
  actionDangerDefaultBorderHover: chroma(colorPalettePrimary.red400).alpha(0.72).hex(),
  actionDangerDefaultBorderPress: chroma(colorPalettePrimary.red400).alpha(0.6).hex(),
  actionDangerDefaultTextDefault: colorPalettePrimary.red400,
  actionDangerDefaultTextHover: chroma(colorPalettePrimary.red400).alpha(0.72).hex(),
  actionDangerDefaultTextPress: chroma(colorPalettePrimary.red400).alpha(0.6).hex(),
  actionDefaultBackgroundDefault: chroma(colorPalettePrimary.blue400).alpha(0).hex(),
  actionDefaultBackgroundHover: chroma(colorPalettePrimary.blue400).alpha(0.08).hex(),
  actionDefaultBackgroundPress: chroma(colorPalettePrimary.blue400).alpha(0.16).hex(),
  actionDefaultBorderDefault: chroma(colorPalettePrimary.white).alpha(0.6).hex(),
  actionDefaultBorderFocus: chroma(colorPalettePrimary.blue400).alpha(0.84).hex(),
  actionDefaultBorderHover: chroma(colorPalettePrimary.blue400).alpha(0.8).hex(),
  actionDefaultBorderPress: chroma(colorPalettePrimary.blue400).alpha(0.76).hex(),
  actionDefaultTextDefault: chroma(colorPalettePrimary.white).alpha(0.84).hex(),
  actionDefaultTextHover: chroma(colorPalettePrimary.blue400).alpha(0.8).hex(),
  actionDefaultTextPress: chroma(colorPalettePrimary.blue400).alpha(0.76).hex(),
  actionDisabledBackground: chroma(colorPalettePrimary.white).alpha(0.24).hex(),
  actionDisabledText: chroma(colorPalettePrimary.white).alpha(0.4).hex(),
  actionPrimaryBackgroundDefault: chroma(colorPalettePrimary.blue400).alpha(0.84).hex(),
  actionPrimaryBackgroundHover: chroma(colorPalettePrimary.blue400).alpha(0.8).hex(),
  actionPrimaryBackgroundPress: chroma(colorPalettePrimary.blue400).alpha(0.76).hex(),
  actionPrimaryTextDefault: colorPalettePrimary.grey800,
  actionPrimaryTextHover: colorPalettePrimary.grey800,
  actionPrimaryTextPress: colorPalettePrimary.grey800,
  actionTertiaryBackgroundDefault: chroma(colorPalettePrimary.blue400).alpha(0).hex(),
  actionTertiaryBackgroundHover: chroma(colorPalettePrimary.blue400).alpha(0.08).hex(),
  actionTertiaryBackgroundPress: chroma(colorPalettePrimary.blue400).alpha(0.16).hex(),
  actionTertiaryTextDefault: chroma(colorPalettePrimary.blue400).alpha(0.84).hex(),
  actionTertiaryTextHover: chroma(colorPalettePrimary.blue400).alpha(0.8).hex(),
  actionTertiaryTextPress: chroma(colorPalettePrimary.blue400).alpha(0.76).hex(),
  backgroundPrimary: colorPalettePrimary.grey800,
  backgroundSecondary: '#283035',
  backgroundValidationDanger: 'transparent',
  backgroundValidationSuccess: 'transparent',
  backgroundValidationWarning: 'transparent',
  border: chroma(colorPalettePrimary.white).alpha(0.48).hex(),
  borderDecorative: chroma(colorPalettePrimary.white).alpha(0.24).hex(),
  borderValidationDanger: colorPalettePrimary.red300,
  borderValidationWarning: colorPalettePrimary.yellow300,
  textPrimary: chroma(colorPalettePrimary.white).alpha(0.84).hex(),
  textSecondary: chroma(colorPalettePrimary.white).alpha(0.6).hex(),
  textPlaceholder: chroma(colorPalettePrimary.grey400).alpha(0.84).hex(),
  textValidationDanger: chroma(colorPalettePrimary.red400).alpha(0.84).hex(),
  textValidationSuccess: chroma(colorPalettePrimary.green400).alpha(0.84).hex(),
  textValidationWarning: chroma(colorPalettePrimary.yellow400).alpha(0.84).hex(),
  textValidationInfo: chroma(colorPalettePrimary.white).alpha(0.6).hex(),
  overlayOverlay: chroma(colorPalettePrimary.grey800).alpha(0.72).hex(),
  // Tags
  tagDefault: chroma(colorPalettePrimary.white).alpha(0.16).hex(),
  tagBrown: chroma(colorPaletteSecondary.brown).alpha(0.84).hex(),
  tagCoral: chroma(colorPaletteSecondary.coral).alpha(0.84).hex(),
  tagCharcoal: chroma(colorPalettePrimary.grey600).alpha(0.84).hex(),
  tagIndigo: chroma(colorPaletteSecondary.indigo).alpha(0.84).hex(),
  tagLemon: chroma(colorPaletteSecondary.lemon).alpha(0.84).hex(),
  tagLime: chroma(colorPaletteSecondary.lime).alpha(0.84).hex(),
  tagPink: chroma(colorPaletteSecondary.pink).alpha(0.84).hex(),
  tagPurple: chroma(colorPaletteSecondary.purple).alpha(0.84).hex(),
  tagTeal: chroma(colorPaletteSecondary.teal).alpha(0.84).hex(),
  tagTurquoise: chroma(colorPaletteSecondary.turquoise).alpha(0.84).hex(),
  tagText: chroma(colorPalettePrimary.white).alpha(0.84).hex(),
  tagHover: chroma(colorPalettePrimary.blue400).alpha(0.08).hex(),
  tagPress: chroma(colorPalettePrimary.blue400).alpha(0.16).hex(),
  tagIconHover: chroma(colorPalettePrimary.white).alpha(0.8).hex(),
  tagIconPress: chroma(colorPalettePrimary.white).alpha(0.76).hex(),
  // Typography
  typographyCodeBg: chroma(colorPalettePrimary.white).alpha(0.16).hex(),
  // Table
  tableSeparatorColor: chroma(colorPalettePrimary.white).alpha(0.24).hex(),
  tableRowHover: chroma(colorPalettePrimary.white).alpha(0.16).hex(),
  tooltipBackgroundTooltip: chroma(colorPalettePrimary.white).alpha(0.6).hex(),
  // Missing in list
  backgroundDanger: 'rgba(200,45,76,0.08)',
  backgroundWarning: 'rgba(222,121,33,0.08)'
};

const darkColors = {
  ...darkColorList,
  ...colorPalettePrimary,
  ...colorPaletteSecondary
};
const lightColors = {
  ...lightColorList,
  ...colorPalettePrimary,
  ...colorPaletteSecondary
};
function getSemanticColors(isDarkMode) {
  return isDarkMode ? darkColors : lightColors;
}

// TODO (FEINF-1674): Remove when removing USE_NEW_RADIO_STYLES
/*
Colors that need to be replaced by semantic colors.
 */
const deprecatedColors = {
  // Radio Colors
  radioInteractiveAvailable: colorPalettePrimary.primary,
  radioInteractiveHover: '#186099',
  radioInteractivePress: '#0D4F85',
  radioDisabled: '#A2AEB8',
  radioDefaultBorder: '#64727D',
  radioDefaultBackground: '#FFFFFF',
  radioInteractiveHoverSecondary: 'rgba(34, 115, 181, 0.08)',
  // Fade of Interactive Available
  radioInteractivePressSecondary: 'rgba(34, 115, 181, 0.16)' // Fade of Interactive Available
};

function getColors(isDarkMode) {
  return {
    ...deprecatedColors,
    ...getSemanticColors(isDarkMode)
  };
}

// eslint-disable-next-line import/no-anonymous-default-export
var antdVars = {
  // IMPORTANT: Do not read this directly from components. Use `React.useContext`.
  'ant-prefix': 'du-bois'
};

const DEFAULT_SPACING_UNIT = 8;
const MODAL_PADDING = 40;
const spacing = {
  xs: DEFAULT_SPACING_UNIT / 2,
  sm: DEFAULT_SPACING_UNIT,
  md: DEFAULT_SPACING_UNIT * 2,
  lg: DEFAULT_SPACING_UNIT * 3
};

// Less variables that are used by AntD
({
  defaultPaddingLg: spacing.lg,
  defaultPaddingMd: spacing.md,
  defaultPaddingSm: spacing.sm,
  defaultPaddingXs: spacing.sm,
  defaultPaddingXss: spacing.xs,
  paddingLg: spacing.md,
  // TODO: Check if there is a better alternative with team
  paddingMd: spacing.sm,
  paddingSm: spacing.sm,
  paddingXs: spacing.xs,
  paddingXss: 0,
  marginSm: 12,
  marginLg: spacing.lg,
  // Button
  btnPaddingHorizontalBase: DEFAULT_SPACING_UNIT * 2,
  btnPaddingHorizontalLg: DEFAULT_SPACING_UNIT * 2,
  btnPaddingHorizontalSm: DEFAULT_SPACING_UNIT * 2,
  // Input
  inputPaddingHorizontal: DEFAULT_SPACING_UNIT * 1.5,
  inputPaddingHorizontalBase: DEFAULT_SPACING_UNIT * 1.5,
  inputPaddingHorizontalSm: DEFAULT_SPACING_UNIT * 1.5,
  inputPaddingHorizontalLg: DEFAULT_SPACING_UNIT * 1.5,
  inputPaddingVertical: spacing.xs + 1,
  inputPaddingVerticalBase: spacing.xs + 1,
  inputPaddingVerticalSm: spacing.xs + 1,
  inputPaddingVerticalLg: spacing.xs + 1,
  // Modal
  modalPadding: MODAL_PADDING,
  modalLessPadding: MODAL_PADDING - 20,
  modalHeaderPadding: `${MODAL_PADDING}px ${MODAL_PADDING}px ${MODAL_PADDING - 20}px`,
  modalHeaderCloseSize: MODAL_PADDING * 2 + 22,
  modalHeaderBorderWidth: 0,
  modalBodyPadding: `0 ${MODAL_PADDING}px`,
  modalFooterPaddingVertical: 0,
  modalFooterPaddingHorizontal: 0,
  modalFooterBorderWidth: 0,
  // Switch
  switchPadding: 0,
  switchHeight: 16,
  switchMinWidth: 28,
  switchPinSize: 14
});
var spacing$1 = spacing;

const heightBase = 40;
const borderWidth = 1;
const antdGeneralVariables = {
  classnamePrefix: antdVars['ant-prefix'],
  iconfontCssPrefix: 'anticon',
  borderRadiusBase: 4,
  borderWidth: borderWidth,
  heightSm: 32,
  heightBase: heightBase,
  iconSize: 24,
  // TODO (FEINF-1545): Update with icon overhaul to 16px
  iconFontSize: 13,
  iconFontSizeNew: 16,
  buttonHeight: heightBase,
  // Height available within button (for label and icon). Same for middle and small buttons.
  buttonInnerHeight: heightBase - spacing$1.sm * 2 - borderWidth * 2
};
const getShadowVariables = isDarkMode => {
  if (isDarkMode) {
    return {
      shadowLow: '0px 4px 16px rgba(0, 0, 0, 0.12)',
      shadowHigh: '0px 8px 24px rgba(0, 0, 0, 0.2);'
    };
  } else {
    return {
      shadowLow: '0px 4px 16px rgba(31, 39, 45, 0.12)',
      shadowHigh: '0px 8px 24px rgba(31, 39, 45, 0.2)'
    };
  }
};
var generalVariables = antdGeneralVariables;

const FONT_SIZE_BASE = 13;

// Less variables that are used by AntD
const antdTypography = {
  fontSizeSm: 12,
  fontSizeBase: FONT_SIZE_BASE,
  fontSizeMd: FONT_SIZE_BASE,
  fontSizeLg: 18,
  fontSizeXl: 22,
  fontSizeXxl: 32,
  lineHeightSm: '16px',
  lineHeightBase: '20px',
  lineHeightMd: '20px',
  lineHeightLg: '24px',
  lineHeightXl: '28px',
  lineHeightXxl: '40px',
  typographyBoldFontWeight: 600
};

// eslint-disable-next-line import/no-anonymous-default-export
var typography = {
  ...antdTypography
};

const defaultOptions = {
  enableAnimation: false,
  zIndexBase: 1000
};

// Function to get variables for a certain theme.
// End users should use useDesignSystemTheme instead.
function getTheme(isDarkMode) {
  let options = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : defaultOptions;
  return {
    colors: getColors(isDarkMode),
    spacing: spacing$1,
    general: {
      ...generalVariables,
      ...getShadowVariables(isDarkMode)
    },
    typography,
    borders: borders$1,
    isDarkMode,
    themeName: isDarkMode ? 'dark' : 'default',
    options
  };
}

function getClassNamePrefix(theme) {
  const antdThemeName = theme.isDarkMode ? 'dark' : 'light';
  return `${theme.general.classnamePrefix}-${antdThemeName}`;
}
function getPrefixedClassNameFromTheme(theme, className) {
  return [getClassNamePrefix(theme), className].filter(Boolean).join('-');
}
function useDesignSystemTheme() {
  const emotionTheme = useTheme();
  // Graceful fallback to default theme in case a test or developer forgot context.
  const theme = emotionTheme && emotionTheme.general ? emotionTheme : getTheme(false);
  return {
    theme: theme,
    classNamePrefix: getClassNamePrefix(theme),
    getPrefixedClassName: className => getPrefixedClassNameFromTheme(theme, className)
  };
}
// This is a simple typed HOC wrapper around the useDesignSystemTheme hook, for use in older react components.
function WithDesignSystemThemeHoc(Component) {
  return function WrappedWithDesignSystemTheme(props) {
    const themeValues = useDesignSystemTheme();
    return jsx(Component, {
      ...props,
      designSystemThemeApi: themeValues
    });
  };
}

const DuboisContextDefaults = {
  enableAnimation: false,
  isDarkMode: false,
  theme: 'default',
  // Prefer to use useDesignSystemTheme.getPrefixedClassName instead
  getPrefixCls: suffix => suffix ? `du-bois-${suffix}` : 'du-bois',
  flags: {}
};
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
const DesignSystemProvider = _ref => {
  let {
    children,
    isDarkMode = false,
    theme: themeName,
    enableAnimation = false,
    zIndexBase = 1000,
    getPopupContainer,
    flags = {}
  } = _ref;
  const theme = useMemo(() => getTheme(isDarkMode, {
    enableAnimation,
    zIndexBase
  }),
  // TODO: revisit this
  // eslint-disable-next-line react-hooks/exhaustive-deps
  [themeName, isDarkMode, zIndexBase]);
  const providerPropsContext = useMemo(() => ({
    isDarkMode,
    theme: themeName,
    enableAnimation,
    zIndexBase,
    getPopupContainer,
    flags
  }), [themeName, isDarkMode, enableAnimation, zIndexBase, getPopupContainer, flags]);
  const classNamePrefix = getClassNamePrefix(theme);
  const value = useMemo(() => {
    return {
      enableAnimation,
      isDarkMode,
      theme: theme.themeName,
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
const ApplyDesignSystemContextOverrides = _ref2 => {
  let {
    isDarkMode,
    theme,
    enableAnimation,
    zIndexBase,
    getPopupContainer,
    flags,
    children
  } = _ref2;
  const parentDesignSystemProviderProps = useContext(DesignSystemProviderPropsContext);
  if (parentDesignSystemProviderProps === null) {
    throw new Error(`ApplyDesignSystemContextOverrides cannot be used standalone - DesignSystemProvider must exist in the React context`);
  }
  const newProps = useMemo(() => ({
    ...parentDesignSystemProviderProps,
    isDarkMode: isDarkMode !== null && isDarkMode !== void 0 ? isDarkMode : parentDesignSystemProviderProps.isDarkMode,
    theme: theme !== null && theme !== void 0 ? theme : parentDesignSystemProviderProps.theme,
    enableAnimation: enableAnimation !== null && enableAnimation !== void 0 ? enableAnimation : parentDesignSystemProviderProps.enableAnimation,
    zIndexBase: zIndexBase !== null && zIndexBase !== void 0 ? zIndexBase : parentDesignSystemProviderProps.zIndexBase,
    getPopupContainer: getPopupContainer !== null && getPopupContainer !== void 0 ? getPopupContainer : parentDesignSystemProviderProps.getPopupContainer,
    flags: {
      ...parentDesignSystemProviderProps.flags,
      ...flags
    }
  }), [parentDesignSystemProviderProps, isDarkMode, theme, enableAnimation, zIndexBase, getPopupContainer, flags]);
  return jsx(DesignSystemProvider, {
    ...newProps,
    children: children
  });
};

// This is a more-specific version of `ApplyDesignSystemContextOverrides` that only allows overriding the flags.
const ApplyDesignSystemFlags = _ref3 => {
  let {
    flags,
    children
  } = _ref3;
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
const DesignSystemAntDConfigProvider = _ref4 => {
  let {
    children
  } = _ref4;
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
const RestoreAntDDefaultClsPrefix = _ref5 => {
  let {
    children
  } = _ref5;
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
      outlineColor: theme.themeName === 'dark' ? theme.colors.actionDefaultBorderFocus : theme.colors.primary
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
        outlineColor: theme.themeName === 'dark' ? theme.colors.actionDefaultBorderFocus : theme.colors.primary
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
function Paragraph(userProps) {
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
      fontWeight: theme.typography.typographyBoldFontWeight,
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
function Title(userProps) {
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
  Typography.Title = Title;
  Typography.Paragraph = Paragraph;
  Typography.Link = Link;
  return Typography;
})();

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
        marginRight: theme.spacing.sm
      },
      [suffixIcon]: {
        marginLeft: theme.spacing.sm
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
    [`& > ${inputClass}`]: {
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
const Group = _ref5 => {
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
  Group
});
const Input = InputNamespace;

// TODO: I'm doing this to support storybook's docgen;
// We should remove this once we have a better storybook integration,
// since these will be exposed in the library's exports.
const __INTERNAL_DO_NOT_USE__TextArea = TextArea;
const __INTERNAL_DO_NOT_USE__Password = Password;
const __INTERNAL_DO_NOT_USE_DEDUPE__Group = Group;

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
    animationDelay: `${delay}s`
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

export { ApplyDesignSystemContextOverrides as A, Button as B, ChevronDownIcon as C, DesignSystemAntDConfigProvider as D, SortUnsortedIcon as E, __INTERNAL_DO_NOT_USE__TextArea as F, __INTERNAL_DO_NOT_USE__Password as G, __INTERNAL_DO_NOT_USE_DEDUPE__Group as H, Icon as I, LoadingIcon as L, NewWindowIcon as N, RestoreAntDDefaultClsPrefix as R, Spinner as S, Typography as T, WithDesignSystemThemeHoc as W, XCircleFillIcon as X, __INTERNAL_DO_NOT_USE__Group as _, useDesignSystemFlags as a, CloseIcon as b, useDesignSystemContext as c, CheckIcon as d, Tooltip as e, getValidationStateColor as f, getAnimationCss as g, Input as h, importantify as i, Checkbox as j, Title as k, DU_BOIS_ENABLE_ANIMATION_CLASSNAME as l, getDefaultStyles as m, getPrimaryStyles as n, getDisabledStyles as o, lightColorList as p, getButtonEmotionStyles as q, getWrapperStyle as r, DesignSystemContext as s, DesignSystemProvider as t, useDesignSystemTheme as u, ApplyDesignSystemFlags as v, useAntDConfigProviderContext as w, SearchIcon as x, SortAscendingIcon as y, SortDescendingIcon as z };
//# sourceMappingURL=Spinner-b67df8d4.js.map
