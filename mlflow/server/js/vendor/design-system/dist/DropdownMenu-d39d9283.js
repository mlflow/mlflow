import React__default, { createContext, useContext, useMemo, useEffect, useLayoutEffect, forwardRef, useRef, useImperativeHandle, Children } from 'react';
import { jsx, Fragment, jsxs } from '@emotion/react/jsx-runtime';
import { useTheme, ThemeProvider, css, keyframes } from '@emotion/react';
import { notification, ConfigProvider, Button as Button$1, Tooltip as Tooltip$1, Input as Input$1, Typography as Typography$1 } from 'antd';
import * as DropdownMenu$1 from '@radix-ui/react-dropdown-menu';
import chroma from 'chroma-js';
import _isNil from 'lodash/isNil';
import _endsWith from 'lodash/endsWith';
import _isBoolean from 'lodash/isBoolean';
import _isNumber from 'lodash/isNumber';
import _isString from 'lodash/isString';
import _mapValues from 'lodash/mapValues';
import unitless from '@emotion/unitless';
import AntDIcon from '@ant-design/icons';
import _uniqueId from 'lodash/uniqueId';

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

function _EMOTION_STRINGIFIED_CSS_ERROR__() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
var _ref2 = process.env.NODE_ENV === "production" ? {
  name: "g8zzui",
  styles: "cursor:progress"
} : {
  name: "4y1qki-AccessibleContainer",
  styles: "cursor:progress;label:AccessibleContainer;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__
};
function AccessibleContainer(_ref) {
  let {
    children,
    label
  } = _ref;
  if (!label) {
    return jsx(Fragment, {
      children: children
    });
  }
  return jsxs("div", {
    css: _ref2,
    children: [jsx("span", {
      css: visuallyHidden,
      children: label
    }), jsx("div", {
      "aria-hidden": true,
      children: children
    })]
  });
}

// Border variables
const borders = {
  borderRadiusMd: 4
};

// eslint-disable-next-line import/no-anonymous-default-export
var borders$1 = {
  ...borders
};

/**
 * These values are based on AntD's breakpoints which follow BootStrap 4 media query rules.
 * The numerical values represent the min-width of the given size.
 * AntD values: https://ant.design/components/grid#col
 * Bootstrap: https://getbootstrap.com/docs/4.0/layout/overview/#responsive-breakpoints
 */

const breakpoints = {
  xs: 0,
  sm: 576,
  md: 768,
  lg: 992,
  xl: 1200,
  xxl: 1600
};
const mediaQueries = {
  xs: '@media (max-width: 575.98px)',
  sm: `@media (min-width: ${breakpoints.sm}px)`,
  md: `@media (min-width: ${breakpoints.md}px)`,
  lg: `@media (min-width: ${breakpoints.lg}px)`,
  xl: `@media (min-width: ${breakpoints.xl}px)`,
  xxl: `@media (min-width: ${breakpoints.xxl}px)`
};
const responsive = {
  breakpoints,
  mediaQueries
};
var responsive$1 = responsive;

/**
 * Provides access to http://go/safex flags from the frontend. Note that this is a temporary
 * workaround until direct `safex` imports are available.
 * @param flag The name of the flag to check
 * @param defaultValue The default value to return if the flag is not set
 * @returns
 */
const safex = (flag, defaultValue) => {
  // Catching errors here, because we don't have type-safety to ensure
  // that `__debug__safex`'s API hasn't changed.
  try {
    const globalSafex = window.__debug__safex;
    return globalSafex ? globalSafex(flag, defaultValue) : defaultValue;
  } catch (e) {
    return defaultValue;
  }
};

// Exported from go/designsystem/colorsheet
const colorPalettePrimary = {
  primary: '#2272B4',
  blue100: '#F0F8FF',
  blue200: '#D7EDFE',
  blue300: '#BAE1FC',
  blue400: '#8FCDFF',
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
const newGreys = {
  grey100: '#F8F8F8',
  grey200: '#E8E8E8',
  grey300: '#D0D0D0',
  grey400: '#AFAFAF',
  grey500: '#969696',
  grey600: '#727272',
  grey700: '#403F3F',
  grey800: '#181818'
};
const colorPaletteSecondary = {
  brown: '#A6630C',
  coral: '#C83243',
  charcoal: colorPalettePrimary.grey600,
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
  borderDecorative: colorPalettePrimary.grey200,
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
  tagCharcoal: colorPaletteSecondary.charcoal,
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
  tagCharcoal: chroma(colorPaletteSecondary.charcoal).alpha(0.84).hex(),
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
  tableRowHover: chroma(colorPalettePrimary.white).alpha(0.16).hex(),
  tooltipBackgroundTooltip: chroma(colorPalettePrimary.white).alpha(0.6).hex(),
  // Missing in list
  backgroundDanger: 'rgba(200,45,76,0.08)',
  backgroundWarning: 'rgba(222,121,33,0.08)'
};
const secondaryColorsUsingNewGreys = {
  charcoal: newGreys.grey600
};
const lightColorsUsingNewGreys = {
  actionDefaultBorderDefault: newGreys.grey300,
  actionDefaultTextDefault: newGreys.grey800,
  actionDisabledBackground: newGreys.grey200,
  actionDisabledText: newGreys.grey400,
  backgroundSecondary: newGreys.grey100,
  border: newGreys.grey300,
  borderDecorative: newGreys.grey200,
  textPrimary: newGreys.grey800,
  textSecondary: newGreys.grey600,
  textPlaceholder: newGreys.grey400,
  textValidationInfo: newGreys.grey600,
  overlayOverlay: chroma(newGreys.grey100).alpha(0.72).hex(),
  tagDefault: chroma(newGreys.grey600).alpha(0.08).hex(),
  tagCharcoal: newGreys.grey600,
  typographyCodeBg: chroma(newGreys.grey600).alpha(0.08).hex(),
  tableRowHover: chroma(newGreys.grey600).alpha(0.04).hex(),
  tooltipBackgroundTooltip: newGreys.grey800
};
const darkColorsUsingNewGreys = {
  actionPrimaryTextDefault: newGreys.grey800,
  actionPrimaryTextHover: newGreys.grey800,
  actionPrimaryTextPress: newGreys.grey800,
  backgroundPrimary: newGreys.grey800,
  backgroundSecondary: newGreys.grey700,
  textPlaceholder: chroma(newGreys.grey400).alpha(0.84).hex(),
  overlayOverlay: chroma(newGreys.grey800).alpha(0.72).hex(),
  tagCharcoal: chroma(newGreys.grey600).alpha(0.84).hex()
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
  const enableNewGreys = safex('databricks.fe.designsystem.enableNewGreys', false);
  const colorList = isDarkMode ? darkColors : lightColors;
  if (enableNewGreys) {
    return {
      ...colorList,
      ...newGreys,
      ...secondaryColorsUsingNewGreys,
      ...(isDarkMode ? darkColorsUsingNewGreys : lightColorsUsingNewGreys)
    };
  } else {
    return colorList;
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
  iconFontSize: 16,
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
    responsive: responsive$1,
    isDarkMode,
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
    flags = {},
    // Disable virtualization of legacy AntD components when running tests so that all items are rendered
    disableLegacyAntVirtualization = process.env.NODE_ENV === 'test' ? true : undefined
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
          getPopupContainer,
          virtual: !disableLegacyAntVirtualization
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

const getIconEmotionStyles = theme => {
  return /*#__PURE__*/css({
    fontSize: theme.general.iconFontSize
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
      "aria-hidden": "true",
      css: getIconEmotionStyles(theme),
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

function SvgCheckIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "m15.06 3.56-9.53 9.531L1 8.561 2.06 7.5l3.47 3.47L14 2.5l1.06 1.06Z",
      clipRule: "evenodd"
    })
  });
}
function CheckIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgCheckIcon
  });
}

function SvgChevronLeftIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M7.083 8 10 10.947 8.958 12 5 8l3.958-4L10 5.053 7.083 8Z",
      clipRule: "evenodd"
    })
  });
}
function ChevronLeftIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgChevronLeftIcon
  });
}

function SvgChevronRightIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M8.917 8 6 5.053 7.042 4 11 8l-3.958 4L6 10.947 8.917 8Z",
      clipRule: "evenodd"
    })
  });
}
function ChevronRightIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgChevronRightIcon
  });
}

function SvgCloseIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M6.97 8.03 2 3.06 3.06 2l4.97 4.97L13 2l1.06 1.06-4.969 4.97 4.97 4.97L13 14.06 8.03 9.092l-4.97 4.97L2 13l4.97-4.97Z",
      clipRule: "evenodd"
    })
  });
}
function CloseIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgCloseIcon
  });
}

function SvgCursorIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("g", {
      clipPath: "url(#CursorIcon_svg__a)",
      children: jsx("path", {
        fill: "currentColor",
        fillRule: "evenodd",
        d: "M1.22 1.22a.75.75 0 0 1 .802-.169l13.5 5.25a.75.75 0 0 1-.043 1.413L9.597 9.597l-1.883 5.882a.75.75 0 0 1-1.413.043l-5.25-13.5a.75.75 0 0 1 .169-.802Zm1.847 1.847 3.864 9.937 1.355-4.233a.75.75 0 0 1 .485-.485l4.233-1.355-9.937-3.864Z",
        clipRule: "evenodd"
      })
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "CursorIcon_svg__a",
        children: jsx("path", {
          fill: "#fff",
          d: "M16 0H0v16h16z"
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

function SvgFaceFrownIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsxs("g", {
      fill: "currentColor",
      clipPath: "url(#FaceFrownIcon_svg__a)",
      children: [jsx("path", {
        d: "M6 5.25a.75.75 0 0 0 0 1.5h.007a.75.75 0 0 0 0-1.5H6ZM9.25 6a.75.75 0 0 1 .75-.75h.007a.75.75 0 0 1 0 1.5H10A.75.75 0 0 1 9.25 6ZM10.07 11.12a.75.75 0 0 0 1.197-.903l-.001-.001v-.001l-.003-.003-.005-.006-.015-.02a2.95 2.95 0 0 0-.217-.246 4.717 4.717 0 0 0-.627-.546C9.86 9 9.04 8.584 8 8.584s-1.858.416-2.4.81a4.716 4.716 0 0 0-.795.733 2.87 2.87 0 0 0-.048.06l-.015.019-.005.006-.002.003-.031.044.03-.042a.75.75 0 1 0 1.22.875 3.219 3.219 0 0 1 .529-.485c.375-.273.89-.523 1.517-.523.627 0 1.142.25 1.517.523a3.219 3.219 0 0 1 .529.485l.021.025.002.003Z"
      }), jsx("path", {
        fillRule: "evenodd",
        d: "M8 .583a7.417 7.417 0 1 0 0 14.834A7.417 7.417 0 0 0 8 .583ZM2.083 8a5.917 5.917 0 1 1 11.834 0A5.917 5.917 0 0 1 2.083 8Z",
        clipRule: "evenodd"
      })]
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "FaceFrownIcon_svg__a",
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
function FaceFrownIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgFaceFrownIcon
  });
}

function SvgFaceNeutralIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsxs("g", {
      fill: "currentColor",
      fillRule: "evenodd",
      clipPath: "url(#FaceNeutralIcon_svg__a)",
      clipRule: "evenodd",
      children: [jsx("path", {
        d: "M8 2.083a5.917 5.917 0 1 0 0 11.834A5.917 5.917 0 0 0 8 2.084ZM.583 8a7.417 7.417 0 1 1 14.834 0A7.417 7.417 0 0 1 .583 8Z"
      }), jsx("path", {
        d: "M4.583 10a.75.75 0 0 1 .75-.75h5.334a.75.75 0 1 1 0 1.5H5.333a.75.75 0 0 1-.75-.75ZM5.25 6A.75.75 0 0 1 6 5.25h.007a.75.75 0 0 1 0 1.5H6A.75.75 0 0 1 5.25 6ZM9.25 6a.75.75 0 0 1 .75-.75h.007a.75.75 0 1 1 0 1.5H10A.75.75 0 0 1 9.25 6Z"
      })]
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "FaceNeutralIcon_svg__a",
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
function FaceNeutralIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgFaceNeutralIcon
  });
}

function SvgFaceSmileIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsxs("g", {
      fill: "currentColor",
      clipPath: "url(#FaceSmileIcon_svg__a)",
      children: [jsx("path", {
        fillRule: "evenodd",
        d: "M8 2.083a5.917 5.917 0 1 0 0 11.834A5.917 5.917 0 0 0 8 2.084ZM.583 8a7.417 7.417 0 1 1 14.834 0A7.417 7.417 0 0 1 .583 8Z",
        clipRule: "evenodd"
      }), jsx("path", {
        d: "M4.883 8.733a.75.75 0 0 1 1.048.147l.002.003.021.026a3.216 3.216 0 0 0 .529.485c.375.273.89.523 1.517.523.627 0 1.142-.25 1.517-.523a3.215 3.215 0 0 0 .55-.511.75.75 0 0 1 1.2.9l.029-.042-.03.043-.001.002-.002.002-.005.007-.015.019a3.878 3.878 0 0 1-.217.247c-.144.15-.354.348-.627.546-.54.393-1.359.81-2.399.81s-1.858-.417-2.4-.81a4.716 4.716 0 0 1-.795-.734 2.643 2.643 0 0 1-.048-.059l-.015-.02-.005-.006-.002-.002v-.002h-.002a.75.75 0 0 1 .15-1.05Z"
      }), jsx("path", {
        fillRule: "evenodd",
        d: "M5.25 6A.75.75 0 0 1 6 5.25h.007a.75.75 0 0 1 0 1.5H6A.75.75 0 0 1 5.25 6ZM9.25 6a.75.75 0 0 1 .75-.75h.007a.75.75 0 1 1 0 1.5H10A.75.75 0 0 1 9.25 6Z",
        clipRule: "evenodd"
      })]
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "FaceSmileIcon_svg__a",
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
function FaceSmileIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgFaceSmileIcon
  });
}

function SvgInfoIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M7.25 11V7h1.5v4h-1.5ZM8 4.5A.75.75 0 1 1 8 6a.75.75 0 0 1 0-1.5Z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8Zm8-6.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13Z",
      clipRule: "evenodd"
    })]
  });
}
function InfoIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgInfoIcon
  });
}

function SvgLoadingIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 24 24",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M23.212 12a.788.788 0 0 1-.789-.788 9.57 9.57 0 0 0-.757-3.751 9.662 9.662 0 0 0-5.129-5.129 9.587 9.587 0 0 0-3.749-.755.788.788 0 0 1 0-1.577c1.513 0 2.983.296 4.365.882a11.128 11.128 0 0 1 3.562 2.403 11.157 11.157 0 0 1 3.283 7.927.785.785 0 0 1-.786.788Z",
      clipRule: "evenodd"
    })
  });
}
function LoadingIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgLoadingIcon
  });
}

function SvgNewWindowIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M10 1h5v5h-1.5V3.56L8.53 8.53 7.47 7.47l4.97-4.97H10V1Z"
    }), jsx("path", {
      fill: "currentColor",
      d: "M1 2.75A.75.75 0 0 1 1.75 2H8v1.5H2.5v10h10V8H14v6.25a.75.75 0 0 1-.75.75H1.75a.75.75 0 0 1-.75-.75V2.75Z"
    })]
  });
}
function NewWindowIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgNewWindowIcon
  });
}

let sequentialCounter = 0;
function useStableUid() {
  return useMemo(() => sequentialCounter++, []);
}

/** Props for UI components that support a "loading" state. */

/**
 * A handler for integrating UI components with external latency instrumentation.
 * If provided via `LoadingStateContext`, hooks will be called whenever child
 * components in a "loading" state are mounted and unmounted.
 */

const LoadingStateContext = /*#__PURE__*/createContext(null);
/**
 * Indicates that the containing component is in a "loading" state, i.e. that the UI
 * displayed to the user is semantically equivalent to them seeing a spinner or a
 * loading indicator. This means that the UI is not in its final settled state yet.
 *
 * All components that are in a "loading" state should render a `<LoadingState>`
 * component, preferrably with an appropriate description.
 *
 * By itself, `<LoadingState>` doesn't do anything, but if used within `LoadingStateContext`,
 * it will call the provided `startLoading()`/`endLoading()` hooks when the component
 * is mounted/unmounted, which can be used to integrate existing latency instrumentation.
 */
const LoadingState = _ref => {
  let {
    description = 'Generic UI loading state'
  } = _ref;
  const uid = useStableUid();
  const loadingStateContext = useContext(LoadingStateContext);
  useLayoutEffect(() => {
    // mount
    if (loadingStateContext) {
      loadingStateContext.startLoading(uid, description);
    }
    return () => {
      // unmount
      if (loadingStateContext) {
        loadingStateContext.endLoading(uid);
      }
    };
  }, [uid, description, loadingStateContext]);
  return null;
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
  let animationDuration = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : 1;
  let inheritColor = arguments.length > 4 && arguments[4] !== undefined ? arguments[4] : false;
  const styles = {
    animation: `${rotate} ${animationDuration}s steps(${frameRate}, end) infinite`,
    ...(inheritColor ? {
      color: 'inherit'
    } : {
      color: theme.colors.textSecondary
    }),
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
    label,
    animationDuration,
    inheritColor,
    loading = true,
    loadingDescription = 'Spinner',
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
    jsxs("div", {
      ...props,
      className: wrapperClass,
      children: [loading && jsx(LoadingState, {
        description: loadingDescription
      }), jsx(AccessibleContainer, {
        label: label,
        children: jsx(LoadingIcon, {
          "aria-hidden": "false",
          css: cssSpinner(theme, frameRate, delay, animationDuration, inheritColor),
          className: className
        })
      })]
    })
  );
};

function getDefaultStyles(theme) {
  let loading = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : false;
  const defaultStyles = {
    backgroundColor: theme.colors.actionDefaultBackgroundDefault,
    borderColor: theme.colors.actionDefaultBorderDefault,
    color: theme.colors.actionDefaultTextDefault
  };
  return {
    ...defaultStyles,
    lineHeight: theme.typography.lineHeightBase,
    textDecoration: 'none',
    '&:hover': {
      backgroundColor: theme.colors.actionDefaultBackgroundHover,
      borderColor: theme.colors.actionDefaultBorderHover,
      color: theme.colors.actionDefaultTextHover
    },
    '&:active': {
      backgroundColor: loading ? theme.colors.actionDefaultBackgroundDefault : theme.colors.actionDefaultBackgroundPress,
      borderColor: theme.colors.actionDefaultBorderPress,
      color: theme.colors.actionDefaultTextPress
    }
  };
}
function getPrimaryStyles(theme) {
  const defaultStyles = {
    backgroundColor: theme.colors.actionPrimaryBackgroundDefault,
    borderColor: 'transparent',
    color: theme.colors.actionPrimaryTextDefault
  };
  return {
    ...defaultStyles,
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
  const defaultStyles = {
    backgroundColor: theme.colors.actionTertiaryBackgroundDefault,
    borderColor: theme.colors.actionTertiaryBackgroundDefault,
    color: theme.colors.actionTertiaryTextDefault
  };
  return {
    ...defaultStyles,
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
  const defaultStyles = {
    backgroundColor: theme.colors.actionDangerPrimaryBackgroundDefault,
    borderColor: 'transparent',
    color: theme.colors.white
  };
  return {
    ...defaultStyles,
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
  const defaultStyles = {
    backgroundColor: theme.colors.actionDangerDefaultBackgroundDefault,
    borderColor: theme.colors.actionDangerDefaultBorderDefault,
    color: theme.colors.actionDangerDefaultTextDefault
  };
  return {
    ...defaultStyles,
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
  const defaultStyles = {
    backgroundColor: theme.colors.actionDisabledBackground,
    borderColor: 'transparent',
    color: theme.colors.actionDisabledText
  };
  return {
    ...defaultStyles,
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
  const defaultStyles = {
    backgroundColor: theme.colors.actionTertiaryBackgroundDefault,
    borderColor: 'transparent',
    color: theme.colors.actionDisabledText
  };
  return {
    ...defaultStyles,
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
    fontSize: theme.general.iconFontSize,
    ...(!isFlex && {
      // verticalAlign used by AntD to move icon and label to center
      // TODO(schs): Try to move buttons to flexbox to solve this. Main problem is that flex-inline and inline-block
      //  behave differently (vertically align of multiple buttons is off). See https://codepen.io/qfactor/pen/JXVzBe
      verticalAlign: -4,
      ...(onlyIcon && {
        verticalAlign: -3
      }),
      // verticalAlign used by AntD to move icon and label to center
      // TODO(schs): Try to move buttons to flexbox to solve this. Main problem is that flex-inline and inline-block
      //  behave differently (vertically align of multiple buttons is off). See https://codepen.io/qfactor/pen/JXVzBe
      // Need to make sure not to apply this to regular buttons as it will offset the icons
      ...(!onlyIcon && {
        verticalAlign: -3
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
    color: theme.colors.textSecondary
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
    ...getDefaultStyles(theme, loading),
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
    [clsLoadingIcon]: {
      display: 'none'
    },
    // Loading styles
    ...(loading && {
      '::before': {
        opacity: 0
      },
      cursor: 'default',
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
      [`& > ${clsLoadingIcon} .anticon`]: {
        paddingRight: 0 // to horizontally center icon
      },

      [`> :not(${clsLoadingIcon})`]: {
        // Hide all content except loading icon
        opacity: 0,
        visibility: 'hidden',
        // Add horizontal space for icon
        ...(withIcon && {
          paddingLeft: theme.general.iconFontSize + theme.spacing.xs
        })
      }
    }),
    // Icon styles
    [`> ${clsIcon} + span, > span + ${clsIcon}`]: {
      marginRight: 0,
      marginLeft: theme.spacing.xs
    },
    [`> ${clsIcon}`]: iconCss,
    [`> ${endIconCssSelector}`]: {
      ...iconCss,
      marginLeft: theme.spacing.sm
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
        },
        ...(loading && {
          '&&, &:hover, &:active': {
            backgroundColor: 'transparent'
          }
        })
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
      ...(loading && {
        '&&, &:hover, &:active': {
          backgroundColor: 'transparent'
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
      ...(loading && {
        '&&, &:hover, &:active': {
          backgroundColor: 'transparent',
          borderColor: theme.colors.actionDefaultBorderDefault
        },
        '&[disabled], &[disabled]:hover, &[disabled]:active': {
          backgroundColor: 'transparent',
          borderColor: 'transparent'
        }
      }),
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
      loading,
      loadingDescription = 'Button',
      endIcon,
      onClick,
      dangerouslySetForceIconStyles,
      dangerouslyUseFocusPseudoClass,
      dangerouslyAppendWrapperCss,
      ...props
    } = _ref2;
    const {
      theme,
      classNamePrefix
    } = useDesignSystemTheme();
    const {
      USE_FLEX_BUTTON: isFlex
    } = useDesignSystemFlags();
    const clsEndIcon = getEndIconClsName(theme);
    const loadingCls = `${classNamePrefix}-btn-loading-icon`;
    const buttonRef = useRef(null);
    useImperativeHandle(ref, () => buttonRef.current);

    // Needed to keep backwards compatibility and support existing unit tests
    useEffect(() => {
      if (buttonRef.current) {
        if (loading) {
          buttonRef.current.setAttribute('loading', 'true');
          buttonRef.current.classList.add(`${classNamePrefix}-btn-loading`);
        } else {
          buttonRef.current.setAttribute('loading', 'false');
          buttonRef.current.classList.remove(`${classNamePrefix}-btn-loading`);
        }
      }
    }, [loading, classNamePrefix]);
    const iconOnly = Boolean((props.icon || endIcon) && !children);
    const handleClick = event => {
      if (loading) {
        return;
      }
      onClick === null || onClick === void 0 ? void 0 : onClick(event);
    };
    const loadingSpinner = jsx(Spinner, {
      className: loadingCls,
      animationDuration: 8,
      inheritColor: true,
      label: "loading",
      "aria-label": "loading",
      loadingDescription: loadingDescription,
      css: /*#__PURE__*/css({
        color: 'inherit !important',
        pointerEvents: 'none',
        ...(!iconOnly && !dangerouslySetForceIconStyles && {
          '.anticon': {
            verticalAlign: '-0.2em'
          }
        }),
        '[aria-hidden="true"]': {
          display: 'inline'
        }
      }, process.env.NODE_ENV === "production" ? "" : ";label:loadingSpinner;")
    });
    return jsx(DesignSystemAntDConfigProvider, {
      children: jsx(Button$1, {
        ...props,
        css: getButtonEmotionStyles({
          theme,
          classNamePrefix,
          loading: Boolean(loading),
          withIcon: Boolean(props.icon),
          onlyIcon: iconOnly,
          isAnchor: Boolean(props.href && !type),
          danger: Boolean(props.danger),
          enableAnimation: theme.options.enableAnimation,
          size: size || 'middle',
          type,
          isFlex,
          forceIconStyles: Boolean(dangerouslySetForceIconStyles),
          useFocusPseudoClass: Boolean(dangerouslyUseFocusPseudoClass)
        }),
        href: props.disabled ? undefined : props.href,
        ...dangerouslySetAntdProps,
        onClick: handleClick,
        icon: loading ? loadingSpinner : props.icon,
        ref: buttonRef,
        type: type === 'tertiary' ? 'link' : type,
        children: children && jsxs("span", {
          style: {
            visibility: loading ? 'hidden' : 'visible',
            ...(isFlex && {
              display: 'inline-flex',
              alignItems: 'center'
            }),
            ...dangerouslyAppendWrapperCss
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

function useUniqueId(prefix) {
  return useMemo(() => _uniqueId(prefix), [prefix]);
}

const Tooltip = _ref => {
  let {
    children,
    title,
    placement = 'top',
    dataTestId,
    dangerouslySetAntdProps,
    silenceScreenReader = false,
    useAsLabel = false,
    ...props
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  const id = useUniqueId('dubois-tooltip-component-');
  if (!title) {
    return jsx(React__default.Fragment, {
      children: children
    });
  }
  const titleProps = silenceScreenReader ? {} : {
    'aria-live': 'polite',
    'aria-relevant': 'additions'
  };
  if (dataTestId) {
    // TODO FEINF-1337 - this should turn into data-testid
    titleProps['data-test-id'] = dataTestId;
  }
  const liveTitle = title && /*#__PURE__*/React__default.isValidElement(title) ? /*#__PURE__*/React__default.cloneElement(title, titleProps) : jsx("span", {
    ...titleProps,
    children: title
  });
  const ariaProps = {
    [useAsLabel ? 'aria-labelledby' : 'aria-describedby']: id,
    'aria-hidden': false
  };
  const childWithProps = /*#__PURE__*/React__default.isValidElement(children) ? /*#__PURE__*/React__default.cloneElement(children, {
    ...ariaProps,
    ...children.props
  }) : _isNil(children) ? children : jsx("span", {
    ...ariaProps,
    children: children
  });
  const {
    overlayInnerStyle,
    overlayStyle,
    ...delegatedDangerouslySetAntdProps
  } = dangerouslySetAntdProps || {};
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Tooltip$1, {
      id: id,
      title: liveTitle,
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
      children: childWithProps
    })
  });
};

const getInputEmotionStyles = (clsPrefix, theme, _ref, useTransparent) => {
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
      },
      '&::placeholder': {
        color: theme.colors.textPlaceholder
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
    [clearIcon]: {
      fontSize: theme.general.iconFontSize
    },
    [prefixIcon]: {
      marginRight: theme.spacing.sm,
      color: theme.colors.textSecondary
    },
    [suffixIcon]: {
      marginLeft: theme.spacing.sm,
      color: theme.colors.textSecondary
    },
    ...getAnimationCss(theme.options.enableAnimation)
  };
  return /*#__PURE__*/css(importantify(styles), process.env.NODE_ENV === "production" ? "" : ";label:getInputEmotionStyles;");
};
const getInputGroupStyling = (clsPrefix, theme) => {
  const inputClass = `.${clsPrefix}-input`;
  const buttonClass = `.${clsPrefix}-btn`;
  return /*#__PURE__*/css({
    display: 'inline-flex !important',
    width: 'auto',
    [`& > ${inputClass}`]: {
      flexGrow: 1,
      '&:disabled': {
        border: 'none',
        '&:hover': {
          borderRight: `1px solid ${theme.colors.actionDisabledText} !important`
        }
      }
    },
    [`& > ${buttonClass} > span`]: {
      verticalAlign: 'middle'
    },
    [`& > ${buttonClass}:disabled, & > ${buttonClass}:disabled:hover`]: {
      borderLeft: `1px solid ${theme.colors.actionDisabledText} !important`
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
    USE_TRANSPARENT_INPUT: useTransparent
  } = useDesignSystemFlags();
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Input$1.TextArea, {
      ref: ref,
      autoComplete: autoComplete,
      css: [getInputEmotionStyles(classNamePrefix, theme, {
        validationState
      }, useTransparent), dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:TextArea;"],
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
    USE_TRANSPARENT_INPUT: useTransparent
  } = useDesignSystemFlags();
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Input$1.Password, {
      visibilityToggle: false,
      ref: ref,
      autoComplete: autoComplete,
      css: [getInputEmotionStyles(classNamePrefix, theme, {
        validationState
      }, useTransparent), dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:Password;"],
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
    onChange,
    onClear,
    ...props
  } = _ref4;
  const {
    classNamePrefix,
    theme
  } = useDesignSystemTheme();
  const {
    USE_TRANSPARENT_INPUT: useTransparent
  } = useDesignSystemFlags();
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
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Input$1, {
      autoComplete: autoComplete,
      ref: ref,
      css: [getInputEmotionStyles(classNamePrefix, theme, {
        validationState
      }, useTransparent), dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:DuboisInput;"],
      onChange: handleChange,
      ...props,
      ...dangerouslySetAntdProps
    })
  });
});
const Group$1 = _ref5 => {
  let {
    dangerouslySetAntdProps,
    dangerouslyAppendEmotionCSS,
    compact = true,
    ...props
  } = _ref5;
  const {
    classNamePrefix,
    theme
  } = useDesignSystemTheme();
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Input$1.Group, {
      css: [getInputGroupStyling(classNamePrefix, theme), dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:Group;"],
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
  Group: Group$1
});
const Input = InputNamespace;

// TODO: I'm doing this to support storybook's docgen;
// We should remove this once we have a better storybook integration,
// since these will be exposed in the library's exports.
const __INTERNAL_DO_NOT_USE__TextArea = TextArea;
const __INTERNAL_DO_NOT_USE__Password = Password;
const __INTERNAL_DO_NOT_USE_DEDUPE__Group = Group$1;

const {
  Text: AntDText$1
} = Typography$1;
function getTextEmotionStyles$1(theme, props) {
  return /*#__PURE__*/css({
    '&&': {
      display: 'block',
      fontSize: theme.typography.fontSizeSm,
      lineHeight: theme.typography.lineHeightSm,
      color: theme.colors.textSecondary,
      ...(props.withoutMargins && {
        '&&': {
          marginTop: 0,
          marginBottom: 0
        }
      })
    }
  }, process.env.NODE_ENV === "production" ? "" : ";label:getTextEmotionStyles;");
}
function Hint(userProps) {
  const {
    dangerouslySetAntdProps,
    bold,
    withoutMargins,
    color,
    ...props
  } = userProps;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(AntDText$1, {
      ...props,
      css: getTextEmotionStyles$1(theme, userProps),
      ...dangerouslySetAntdProps
    })
  });
}

const getLinkStyles = (theme, clsPrefix) => {
  const centerIcons = safex('databricks.fe.designsystem.centerIconsInText', false);
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
    '.anticon': {
      fontSize: 12,
      ...(centerIcons ? {
        verticalAlign: 'baseline'
      } : {})
    }
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
const Link = /*#__PURE__*/forwardRef(function Link(_ref, ref) {
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
  const linkStyles = props.ellipsis && openInNewTab ? [getLinkStyles(theme, classNamePrefix), getEllipsisNewTabLinkStyles()] : getLinkStyles(theme, classNamePrefix);
  const iconStyles = props.ellipsis ? [getIconStyles(theme), getEllipsisIconStyles()] : getIconStyles(theme);
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsxs(Typography$1.Link, {
      "aria-disabled": linkProps.disabled,
      css: linkStyles,
      ref: ref,
      ...linkProps,
      ...dangerouslySetAntdProps,
      children: [children, openInNewTab ? jsx(NewWindowIcon, {
        css: iconStyles,
        ...newTabProps
      }) : null]
    })
  });
});

const {
  Paragraph: AntDParagraph
} = Typography$1;
function getParagraphEmotionStyles(theme, props) {
  const centerIcons = safex('databricks.fe.designsystem.centerIconsInText', false);
  return /*#__PURE__*/css({
    '&&': {
      fontSize: theme.typography.fontSizeBase,
      lineHeight: theme.typography.lineHeightBase,
      color: getTypographyColor(theme, props.color, theme.colors.textPrimary)
    },
    ...(centerIcons && {
      '& .anticon': {
        verticalAlign: 'text-bottom'
      }
    })
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
  const centerIcons = safex('databricks.fe.designsystem.centerIconsInText', false);
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
            lineHeight: theme.typography.lineHeightXxl,
            ...(centerIcons && {
              '& .anticon': {
                lineHeight: theme.typography.lineHeightXxl,
                verticalAlign: 'middle'
              }
            })
          };
        case 'xl':
          return {
            fontSize: theme.typography.fontSizeXl,
            lineHeight: theme.typography.lineHeightXl,
            ...(centerIcons && {
              '& .anticon': {
                lineHeight: theme.typography.lineHeightXl,
                verticalAlign: 'middle'
              }
            })
          };
        case 'lg':
          return {
            fontSize: theme.typography.fontSizeLg,
            lineHeight: theme.typography.lineHeightLg,
            ...(centerIcons && {
              '& .anticon': {
                lineHeight: theme.typography.lineHeightLg,
                verticalAlign: 'middle'
              }
            })
          };
        case 'sm':
          return {
            fontSize: theme.typography.fontSizeSm,
            lineHeight: theme.typography.lineHeightSm,
            ...(centerIcons && {
              '& .anticon': {
                verticalAlign: '-0.219em'
              }
            })
          };
        default:
          return {};
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
  const centerIcons = safex('databricks.fe.designsystem.centerIconsInText', false);
  switch (props.level) {
    case 1:
      return /*#__PURE__*/css({
        '&&': {
          fontSize: theme.typography.fontSizeXxl,
          lineHeight: theme.typography.lineHeightXxl,
          fontWeight: theme.typography.typographyBoldFontWeight
        },
        ...(centerIcons && {
          '& > .anticon': {
            lineHeight: theme.typography.lineHeightXxl
          }
        })
      }, process.env.NODE_ENV === "production" ? "" : ";label:getLevelStyles;");
    case 2:
      return /*#__PURE__*/css({
        '&&': {
          fontSize: theme.typography.fontSizeXl,
          lineHeight: theme.typography.lineHeightXl,
          fontWeight: theme.typography.typographyBoldFontWeight
        },
        ...(centerIcons && {
          '& > .anticon': {
            lineHeight: theme.typography.lineHeightXl
          }
        })
      }, process.env.NODE_ENV === "production" ? "" : ";label:getLevelStyles;");
    case 3:
      return /*#__PURE__*/css({
        '&&': {
          fontSize: theme.typography.fontSizeLg,
          lineHeight: theme.typography.lineHeightLg,
          fontWeight: theme.typography.typographyBoldFontWeight
        },
        ...(centerIcons && {
          '& > .anticon': {
            lineHeight: theme.typography.lineHeightLg
          }
        })
      }, process.env.NODE_ENV === "production" ? "" : ";label:getLevelStyles;");
    case 4:
    default:
      return /*#__PURE__*/css({
        '&&': {
          fontSize: theme.typography.fontSizeMd,
          lineHeight: theme.typography.lineHeightMd,
          fontWeight: theme.typography.typographyBoldFontWeight
        },
        ...(centerIcons && {
          '& > .anticon': {
            lineHeight: theme.typography.lineHeightMd
          }
        })
      }, process.env.NODE_ENV === "production" ? "" : ";label:getLevelStyles;");
  }
}
function getTitleEmotionStyles(theme, props) {
  const centerIcons = safex('databricks.fe.designsystem.centerIconsInText', false);
  return /*#__PURE__*/css(getLevelStyles(theme, props), {
    '&&': {
      color: getTypographyColor(theme, props.color, theme.colors.textPrimary)
    },
    ...(centerIcons && {
      '& > .anticon': {
        verticalAlign: 'middle'
      }
    })
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
  Typography.Hint = Hint;
  return Typography;
})();

const Root = DropdownMenu$1.Root; // Behavioral component only

const Content = /*#__PURE__*/forwardRef(function Content(_ref, ref) {
  let {
    children,
    minWidth = 220,
    isInsideModal,
    onEscapeKeyDown,
    onKeyDown,
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
      css: [contentStyles, {
        minWidth
      }, process.env.NODE_ENV === "production" ? "" : ";label:Content;"],
      sideOffset: 4,
      align: "start",
      onKeyDown: e => {
        // This is a workaround for Radix's DropdownMenu.Content not receiving Escape key events
        // when nested inside a modal. We need to stop propagation of the event so that the modal
        // doesn't close when the DropdownMenu should.
        if (e.key === 'Escape') {
          if (isInsideModal) {
            e.stopPropagation();
          }
          onEscapeKeyDown === null || onEscapeKeyDown === void 0 ? void 0 : onEscapeKeyDown(e.nativeEvent);
        }
        onKeyDown === null || onKeyDown === void 0 ? void 0 : onKeyDown(e);
      },
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
      css: [contentStyles, {
        minWidth
      }, process.env.NODE_ENV === "production" ? "" : ";label:SubContent;"],
      sideOffset: -2,
      alignOffset: -5,
      ...props,
      children: children
    })
  });
});
const Trigger = /*#__PURE__*/forwardRef(function Trigger(_ref3, ref) {
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
    css: theme => [dropdownItemStyles, danger && dangerItemStyles(theme)],
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
const Label = /*#__PURE__*/forwardRef(function Label(_ref5, ref) {
  let {
    children,
    ...props
  } = _ref5;
  return jsx(DropdownMenu$1.Label, {
    ref: ref,
    css: [dropdownItemStyles, theme => ({
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
    css: dropdownSeparatorStyles,
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
    css: [dropdownItemStyles, theme => ({
      '&[data-state="open"]': {
        backgroundColor: theme.colors.actionTertiaryBackgroundHover
      }
    }), process.env.NODE_ENV === "production" ? "" : ";label:SubTrigger;"],
    ...props,
    children: [getNewChildren(children, props, disabledReason, subTriggerRef), jsx(HintColumn, {
      css: theme => ({
        margin: CONSTANTS.subMenuIconMargin(theme),
        display: 'flex',
        alignSelf: 'stretch',
        alignItems: 'center'
      }),
      children: jsx(ChevronRightIcon, {
        css: theme => ({
          fontSize: CONSTANTS.subMenuIconSize(theme)
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
  const checkboxItemRef = useRef(null);
  useImperativeHandle(ref, () => checkboxItemRef.current);
  return jsx(DropdownMenu$1.CheckboxItem, {
    ref: checkboxItemRef,
    css: theme => [dropdownItemStyles, checkboxItemStyles(theme)],
    ...props,
    children: getNewChildren(children, props, disabledReason, checkboxItemRef)
  });
});
const ItemIndicator = /*#__PURE__*/forwardRef(function ItemIndicator(_ref9, ref) {
  let {
    children,
    ...props
  } = _ref9;
  return jsx(DropdownMenu$1.ItemIndicator, {
    ref: ref,
    css: theme => ({
      marginLeft: -(CONSTANTS.checkboxIconWidth(theme) + CONSTANTS.checkboxPaddingRight(theme)),
      position: 'absolute',
      fontSize: theme.general.iconFontSize
    }),
    ...props,
    children: children !== null && children !== void 0 ? children : jsx(CheckIcon, {
      css: theme => ({
        color: theme.colors.textSecondary
      })
    })
  });
});
const Arrow = /*#__PURE__*/forwardRef(function Arrow(_ref10, ref) {
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
const RadioItem = /*#__PURE__*/forwardRef(function RadioItem(_ref11, ref) {
  let {
    children,
    disabledReason,
    ...props
  } = _ref11;
  const radioItemRef = useRef(null);
  useImperativeHandle(ref, () => radioItemRef.current);
  return jsx(DropdownMenu$1.RadioItem, {
    ref: radioItemRef,
    css: theme => [dropdownItemStyles, checkboxItemStyles(theme)],
    ...props,
    children: getNewChildren(children, props, disabledReason, radioItemRef)
  });
});

// UNWRAPPED RADIX-UI-COMPONENTS
const Group = DropdownMenu$1.Group;
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
      paddingRight: theme.spacing.sm
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
      children: jsx(InfoIcon, {
        "aria-hidden": "false"
      })
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
const CONSTANTS = {
  itemPaddingVertical(theme) {
    // The number from the mocks is the midpoint between constants
    return 0.5 * theme.spacing.xs + 0.5 * theme.spacing.sm;
  },
  itemPaddingHorizontal(theme) {
    return theme.spacing.sm;
  },
  checkboxIconWidth(theme) {
    return theme.general.iconFontSize;
  },
  checkboxPaddingLeft(theme) {
    return theme.spacing.sm + theme.spacing.xs;
  },
  checkboxPaddingRight(theme) {
    return theme.spacing.sm;
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
      color: theme.colors.textPrimary,
      textDecoration: 'none'
    }
  })
});
const contentStyles = theme => ({
  ...dropdownContentStyles(theme)
});
const dropdownItemStyles = theme => ({
  padding: `${CONSTANTS.itemPaddingVertical(theme)}px ${CONSTANTS.itemPaddingHorizontal(theme)}px`,
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
const checkboxItemStyles = theme => ({
  position: 'relative',
  paddingLeft: CONSTANTS.checkboxIconWidth(theme) + CONSTANTS.checkboxPaddingLeft(theme) + CONSTANTS.checkboxPaddingRight(theme)
});
const metaTextStyles = theme => ({
  color: theme.colors.textSecondary,
  fontSize: theme.typography.fontSizeSm,
  '[data-disabled] &': {
    color: theme.colors.actionDisabledText
  }
});
const dropdownSeparatorStyles = theme => ({
  height: 1,
  margin: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
  backgroundColor: theme.colors.borderDecorative
});

var DropdownMenu = /*#__PURE__*/Object.freeze({
  __proto__: null,
  Arrow: Arrow,
  CheckboxItem: CheckboxItem,
  Content: Content,
  Group: Group,
  HintColumn: HintColumn,
  HintRow: HintRow,
  IconWrapper: IconWrapper,
  Item: Item,
  ItemIndicator: ItemIndicator,
  Label: Label,
  RadioGroup: RadioGroup,
  RadioItem: RadioItem,
  Root: Root,
  Separator: Separator,
  Sub: Sub,
  SubContent: SubContent,
  SubTrigger: SubTrigger,
  Trigger: Trigger,
  TriggerItem: TriggerItem,
  dropdownContentStyles: dropdownContentStyles,
  dropdownItemStyles: dropdownItemStyles,
  dropdownSeparatorStyles: dropdownSeparatorStyles
});

export { dropdownItemStyles as $, ApplyDesignSystemContextOverrides as A, Button as B, CloseIcon as C, DesignSystemAntDConfigProvider as D, DesignSystemThemeContext as E, DesignSystemContext as F, DesignSystemThemeProvider as G, DesignSystemProvider as H, Icon as I, ApplyDesignSystemFlags as J, useAntDConfigProviderContext as K, LoadingState as L, DropdownMenu as M, CursorIcon as N, FaceFrownIcon as O, FaceNeutralIcon as P, FaceSmileIcon as Q, RestoreAntDDefaultClsPrefix as R, Spinner as S, Tooltip as T, NewWindowIcon as U, __INTERNAL_DO_NOT_USE__Password as V, WithDesignSystemThemeHoc as W, __INTERNAL_DO_NOT_USE_DEDUPE__Group as X, LoadingStateContext as Y, getTypographyColor as Z, __INTERNAL_DO_NOT_USE__TextArea as _, ChevronRightIcon as a, dropdownSeparatorStyles as a0, useDesignSystemFlags as b, useDesignSystemContext as c, InfoIcon as d, Input as e, CheckIcon as f, getAnimationCss as g, getValidationStateColor as h, importantify as i, Typography as j, LoadingIcon as k, dropdownContentStyles as l, Title as m, AccessibleContainer as n, ChevronLeftIcon as o, DU_BOIS_ENABLE_ANIMATION_CLASSNAME as p, Root as q, Trigger as r, safex as s, getDefaultStyles as t, useDesignSystemTheme as u, visuallyHidden as v, getPrimaryStyles as w, getDisabledStyles as x, lightColorList as y, getButtonEmotionStyles as z };
//# sourceMappingURL=DropdownMenu-d39d9283.js.map
