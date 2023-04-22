import React__default, { createContext, useMemo, useEffect, useContext, forwardRef } from 'react';
import { jsx, jsxs } from '@emotion/react/jsx-runtime';
import { useTheme, ThemeProvider, css } from '@emotion/react';
import { notification, ConfigProvider, Checkbox as Checkbox$1, Typography as Typography$1 } from 'antd';
import classnames from 'classnames';
import chroma from 'chroma-js';
import AntDIcon from '@ant-design/icons';
import _isNil from 'lodash/isNil';
import _endsWith from 'lodash/endsWith';
import _isBoolean from 'lodash/isBoolean';
import _isNumber from 'lodash/isNumber';
import _isString from 'lodash/isString';
import _mapValues from 'lodash/mapValues';
import unitless from '@emotion/unitless';

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

// Colors taken from figma designs but not part of official list (yet)
const unstableColors = {
  textValidationInfo: '#64727D',
  backgroundValidationDanger: colorPalettePrimary.red100,
  backgroundValidationSuccess: colorPalettePrimary.blue100,
  backgroundValidationWarning: colorPalettePrimary.yellow100,
  borderValidationDanger: colorPalettePrimary.red300,
  borderValidationSuccess: colorPalettePrimary.blue300,
  borderValidationWarning: colorPalettePrimary.yellow300
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
  border: colorPalettePrimary.grey300,
  borderDecorative: colorPalettePrimary.grey300,
  textPrimary: colorPalettePrimary.grey800,
  textSecondary: colorPalettePrimary.grey600,
  textPlaceholder: colorPalettePrimary.grey400,
  textValidationDanger: colorPalettePrimary.red600,
  textValidationSuccess: colorPalettePrimary.green600,
  textValidationWarning: colorPalettePrimary.yellow600,
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
  tableBorder: colorPalettePrimary.grey200,
  tableRowHover: chroma(colorPalettePrimary.grey600).alpha(0.04).hex(),
  tooltipBackgroundTooltip: colorPalettePrimary.grey800,
  ...unstableColors
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
  backgroundSecondary: chroma(colorPalettePrimary.white).alpha(0.04).hex(),
  border: chroma(colorPalettePrimary.white).alpha(0.48).hex(),
  borderDecorative: chroma(colorPalettePrimary.white).alpha(0.24).hex(),
  textPrimary: chroma(colorPalettePrimary.white).alpha(0.84).hex(),
  textSecondary: chroma(colorPalettePrimary.white).alpha(0.6).hex(),
  textPlaceholder: chroma(colorPalettePrimary.grey400).alpha(0.84).hex(),
  textValidationDanger: chroma(colorPalettePrimary.red400).alpha(0.84).hex(),
  textValidationSuccess: chroma(colorPalettePrimary.green400).alpha(0.84).hex(),
  textValidationWarning: chroma(colorPalettePrimary.yellow400).alpha(0.84).hex(),
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
  tableBorder: chroma(colorPalettePrimary.white).alpha(0.24).hex(),
  tableRowHover: chroma(colorPalettePrimary.white).alpha(0.16).hex(),
  tooltipBackgroundTooltip: chroma(colorPalettePrimary.white).alpha(0.6).hex(),
  // Missing in list
  backgroundDanger: 'rgba(200,45,76,0.08)',
  backgroundWarning: 'rgba(222,121,33,0.08)',
  // Missing in light list
  // "background-validation-danger": "rgba(229,110,134,0)",
  // "background-validation-warning": "rgba(252,234,202,0)",
  ...unstableColors
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
function getSemanticColors(theme) {
  switch (theme) {
    case 'dark':
      return {
        ...darkColors
      };
    default:
      return {
        ...lightColors
      };
  }
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

function getColors(theme) {
  return {
    ...deprecatedColors,
    ...getSemanticColors(theme)
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
  modalHeaderPadding: "".concat(MODAL_PADDING, "px ").concat(MODAL_PADDING, "px ").concat(MODAL_PADDING - 20, "px"),
  modalHeaderCloseSize: MODAL_PADDING * 2 + 22,
  modalHeaderBorderWidth: 0,
  modalBodyPadding: "0 ".concat(MODAL_PADDING, "px"),
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

// TODO(giles): fix this
// eslint-disable-next-line import/no-named-as-default

const defaultOptions = {
  enableAnimation: false,
  zIndexBase: 1000
};

// Function to get variables for a certain theme.
// End users should use useDesignSystemTheme instead.
function getTheme(themeName) {
  let options = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : defaultOptions;
  return {
    colors: getColors(themeName),
    spacing: spacing$1,
    general: generalVariables,
    typography,
    borders: borders$1,
    themeName,
    options
  };
}

const themeNameToAntdName = {
  dark: 'dark',
  default: 'light'
};
function getClassNamePrefix(theme) {
  const antdThemeName = themeNameToAntdName[theme.themeName];
  return "".concat(theme.general.classnamePrefix, "-").concat(antdThemeName);
}
function getPrefixedClassNameFromTheme(theme, className) {
  return [getClassNamePrefix(theme), className].filter(Boolean).join('-');
}
function useDesignSystemTheme() {
  const emotionTheme = useTheme();
  // Graceful fallback to default theme in case a test or developer forgot context.
  const theme = emotionTheme && emotionTheme.general ? emotionTheme : getTheme('default');
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
  theme: 'default',
  // Prefer to use useDesignSystemTheme.getPrefixedClassName instead
  getPrefixCls: suffix => suffix ? "du-bois-".concat(suffix) : 'du-bois',
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
    ["[class*=du-bois]:not(.".concat(DU_BOIS_ENABLE_ANIMATION_CLASSNAME, ", .").concat(DU_BOIS_ENABLE_ANIMATION_CLASSNAME, " *)")]: {
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
    theme: themeName,
    enableAnimation = false,
    zIndexBase = 1000,
    getPopupContainer,
    flags = {}
  } = _ref;
  const theme = useMemo(() => getTheme(themeName || 'default', {
    enableAnimation,
    zIndexBase
  }),
  // TODO: revisit this
  // eslint-disable-next-line react-hooks/exhaustive-deps
  [themeName, zIndexBase]);
  const providerPropsContext = useMemo(() => ({
    theme: themeName,
    enableAnimation,
    zIndexBase,
    getPopupContainer,
    flags
  }), [themeName, enableAnimation, zIndexBase, getPopupContainer, flags]);
  const classNamePrefix = getClassNamePrefix(theme);
  const value = useMemo(() => {
    return {
      enableAnimation,
      theme: theme.themeName,
      getPrefixCls: suffix => getPrefixedClassNameFromTheme(theme, suffix),
      getPopupContainer,
      flags
    };
  }, [enableAnimation, theme, getPopupContainer, flags]);
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
    theme,
    enableAnimation,
    zIndexBase,
    getPopupContainer,
    flags,
    children
  } = _ref2;
  const parentDesignSystemProviderProps = useContext(DesignSystemProviderPropsContext);
  if (parentDesignSystemProviderProps === null) {
    throw new Error("ApplyDesignSystemContextOverrides cannot be used standalone - DesignSystemProvider must exist in the React context");
  }
  const newProps = useMemo(() => ({
    ...parentDesignSystemProviderProps,
    theme: theme !== null && theme !== void 0 ? theme : parentDesignSystemProviderProps.theme,
    enableAnimation: enableAnimation !== null && enableAnimation !== void 0 ? enableAnimation : parentDesignSystemProviderProps.enableAnimation,
    zIndexBase: zIndexBase !== null && zIndexBase !== void 0 ? zIndexBase : parentDesignSystemProviderProps.zIndexBase,
    getPopupContainer: getPopupContainer !== null && getPopupContainer !== void 0 ? getPopupContainer : parentDesignSystemProviderProps.getPopupContainer,
    flags: {
      ...parentDesignSystemProviderProps.flags,
      ...flags
    }
  }), [parentDesignSystemProviderProps, theme, enableAnimation, zIndexBase, getPopupContainer, flags]);
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
    throw new Error("ApplyDesignSystemFlags cannot be used standalone - DesignSystemProvider must exist in the React context");
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
          return "".concat(value, "!important");
        }
        return "".concat(value, "px!important");
      }
      return "".concat(value, "!important");
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

function getCheckboxEmotionStyles(clsPrefix, theme) {
  let isHorizontal = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : false;
  let useNewCheckboxStyles = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : false;
  const classInput = ".".concat(clsPrefix, "-input");
  const classInner = ".".concat(clsPrefix, "-inner");
  const classIndeterminate = ".".concat(clsPrefix, "-indeterminate");
  const classChecked = ".".concat(clsPrefix, "-checked");
  const classDisabled = ".".concat(clsPrefix, "-disabled");
  const classDisabledWrapper = ".".concat(clsPrefix, "-wrapper-disabled");
  const classContainer = ".".concat(clsPrefix, "-group");
  const classWrapper = ".".concat(clsPrefix, "-wrapper");
  const defaultSelector = "".concat(classInput, " + ").concat(classInner);
  const hoverSelector = "".concat(classInput, ":hover + ").concat(classInner);
  const pressSelector = "".concat(classInput, ":active + ").concat(classInner);
  const styles = {
    ...(useNewCheckboxStyles && {
      [".".concat(clsPrefix)]: {
        top: 'unset',
        lineHeight: theme.typography.lineHeightBase
      },
      ["&".concat(classWrapper, ", ").concat(classWrapper)]: {
        alignItems: 'center',
        lineHeight: theme.typography.lineHeightBase
      }
    }),
    // Top level styles are for the unchecked state
    [classInner]: {
      borderColor: theme.colors.actionDefaultBorderDefault
    },
    // Layout styling
    ["&".concat(classContainer)]: {
      display: 'flex',
      flexDirection: 'column',
      rowGap: theme.spacing.sm,
      columnGap: 0
    },
    ...(isHorizontal && {
      ["&".concat(classContainer)]: {
        display: 'flex',
        flexDirection: 'row',
        columnGap: theme.spacing.sm,
        rowGap: 0,
        ["& > ".concat(classContainer, "-item")]: {
          marginRight: 0
        }
      }
    }),
    // Keyboard focus
    ["".concat(classInput, ":focus-visible + ").concat(classInner)]: {
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
    ["&".concat(classDisabledWrapper)]: {
      [classDisabled]: {
        // Disabled Checked
        ["&".concat(classChecked)]: {
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
        ["&".concat(classIndeterminate)]: {
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
      ["&& + .".concat(clsPrefix, "-hint, && + .").concat(clsPrefix, "-form-message")]: {
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
      className: classnames(className, "".concat(clsPrefix, "-container")),
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
        children: children
      })
    })
  });
});
const CheckboxGroup = /*#__PURE__*/forwardRef(function CheckboxGroup(_ref3, ref) {
  let {
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
      css: getCheckboxEmotionStyles(clsPrefix, theme, layout === 'horizontal', USE_NEW_CHECKBOX_STYLES)
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
  const classTypography = ".".concat(clsPrefix, "-typography");
  const styles = {
    ["&".concat(classTypography, ", &").concat(classTypography, ":focus")]: {
      color: theme.colors.actionTertiaryTextDefault
    },
    ["&".concat(classTypography, ":hover, &").concat(classTypography, ":hover .anticon")]: {
      color: theme.colors.actionTertiaryTextHover,
      textDecoration: 'underline'
    },
    ["&".concat(classTypography, ":active, &").concat(classTypography, ":active .anticon")]: {
      color: theme.colors.actionTertiaryTextPress,
      textDecoration: 'underline'
    },
    ["&".concat(classTypography, ":focus-visible")]: {
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

export { ApplyDesignSystemContextOverrides as A, Checkbox as C, DesignSystemAntDConfigProvider as D, Icon as I, NewWindowIcon as N, RestoreAntDDefaultClsPrefix as R, SortAscendingIcon as S, Typography as T, WithDesignSystemThemeHoc as W, __INTERNAL_DO_NOT_USE__Group as _, useDesignSystemFlags as a, useDesignSystemContext as b, getValidationStateColor as c, Title as d, DU_BOIS_ENABLE_ANIMATION_CLASSNAME as e, getWrapperStyle as f, getAnimationCss as g, DesignSystemContext as h, importantify as i, DesignSystemProvider as j, ApplyDesignSystemFlags as k, lightColorList as l, useAntDConfigProviderContext as m, SortDescendingIcon as n, SortUnsortedIcon as o, useDesignSystemTheme as u };
//# sourceMappingURL=Typography-84a00e02.js.map
