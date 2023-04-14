import { useTheme } from '@emotion/react';
import 'react';
import chroma from 'chroma-js';
import { jsx } from '@emotion/react/jsx-runtime';

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

export { WithDesignSystemThemeHoc as W, getClassNamePrefix as a, getPrefixedClassNameFromTheme as b, getTheme as g, lightColorList as l, useDesignSystemTheme as u };
//# sourceMappingURL=useDesignSystemTheme-1e16bcf5.js.map
