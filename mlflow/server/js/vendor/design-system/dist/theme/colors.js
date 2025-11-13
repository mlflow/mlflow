import { primitiveColors } from './_generated/PrimitiveColors';
import { darkProtectedColorList, lightProtectedColorList } from './_generated/ProtectedSemanticColors';
import { darkColorList } from './_generated/SemanticColors-Dark';
import { lightColorList } from './_generated/SemanticColors-Light';
// Currently these are the same for both light and dark mode, but we may want to change this in the future.
const branded = {
    ai: {
        /** For AI components, the top-left-oriented start color of gradient treatments. */
        gradientStart: '#4299E0',
        /** For AI components, the mid color of gradient treatments. */
        gradientMid: '#CA42E0',
        /** For AI components, the bottom-right-oriented end color of gradient treatments. */
        gradientEnd: '#FF5F46',
    },
};
const darkColors = { ...darkColorList, ...primitiveColors, branded };
const lightColors = { ...lightColorList, ...primitiveColors, branded };
function getSemanticColors(isDarkMode) {
    return isDarkMode ? darkColors : lightColors;
}
export function getProtectedSemanticColors(isDarkMode) {
    return isDarkMode ? darkProtectedColorList : lightProtectedColorList;
}
// Maps Du Bois colors to `antd` names. This is exclusively used for theming
// Du Bois's wrapped `antd` components.
export function getAntdColors(isDarkMode) {
    const semanticColors = getSemanticColors(isDarkMode);
    return {
        bodyBackground: semanticColors.backgroundPrimary,
        textColor: semanticColors.textPrimary,
        textColorSecondary: semanticColors.textSecondary,
        primaryColor: semanticColors.blue600,
        infoColor: '#64727D',
        errorColor: semanticColors.actionDangerPrimaryBackgroundDefault,
        successColor: '#34824F',
        warningColor: '#AF5B23',
        borderColorBase: semanticColors.border,
        // Alert colors (AntD variables)
        alertErrorTextColor: semanticColors.textValidationDanger,
        alertErrorBorderColor: semanticColors.red300,
        alertErrorBgColor: semanticColors.red100,
        alertErrorIconColor: semanticColors.textValidationDanger,
        alertInfoTextColor: semanticColors.textPrimary,
        alertInfoBorderColor: semanticColors.grey300,
        alertInfoBgColor: semanticColors.backgroundSecondary,
        alertInfoIconColor: semanticColors.textSecondary,
        alertWarningTextColor: semanticColors.textValidationWarning,
        alertWarningBorderColor: semanticColors.yellow300,
        alertWarningBgColor: semanticColors.yellow100,
        alertWarningIconColor: semanticColors.textValidationWarning,
        ...(isDarkMode && {
            alertErrorBgColor: 'transparent',
            alertInfoBgColor: 'transparent',
            alertWarningBgColor: 'transparent',
        }),
        alertTextColor: 'inherit',
        alertMessageColor: 'inherit',
        spinDotDefault: semanticColors.textSecondary,
    };
}
// When deprecating a color, add it to this object with a comment explaining why it's deprecated and a link to a JIRA ticket.
// Example: `@deprecated This color supports XXXX will be removed in an upcoming release (FEINF-1234).`
const deprecatedPrimitiveColors = {
    /** @deprecated This was an alias to `primitiveColors.blue600`, please use that instead.
     * If possible, please use an appropriate semantic color, such as `actionPrimaryBackgroundDefault`. */
    primary: primitiveColors.blue600,
    /** @deprecated This was an alias to `primitiveColors.grey600`, please use that instead.
     * If possible, please use an appropriate semantic color, such as `actionPrimaryBackgroundHover`. */
    charcoal: primitiveColors.grey600,
    /** @deprecated This color supports legacy radio styles and will be removed in an upcoming release (FEINF-1674). */
    radioInteractiveAvailable: primitiveColors.blue600,
    /** @deprecated This color supports legacy radio styles and will be removed in an upcoming release (FEINF-1674). */
    radioInteractiveHover: '#186099',
    /** @deprecated This color supports legacy radio styles and will be removed in an upcoming release (FEINF-1674). */
    radioInteractivePress: '#0D4F85',
    /** @deprecated This color supports legacy radio styles and will be removed in an upcoming release (FEINF-1674). */
    radioDisabled: '#A2AEB8',
    /** @deprecated This color supports legacy radio styles and will be removed in an upcoming release (FEINF-1674). */
    radioDefaultBorder: '#64727D',
    /** @deprecated This color supports legacy radio styles and will be removed in an upcoming release (FEINF-1674). */
    radioDefaultBackground: '#FFFFFF',
    /** @deprecated This color supports legacy radio styles and will be removed in an upcoming release (FEINF-1674). */
    radioInteractiveHoverSecondary: 'rgba(34, 115, 181, 0.08)',
    /** @deprecated This color supports legacy radio styles and will be removed in an upcoming release (FEINF-1674). */
    radioInteractivePressSecondary: 'rgba(34, 115, 181, 0.16)',
};
export const deprecatedSemanticColorsLight = {
    /** @deprecated Use `backgroundDanger` (FEINF-xxxx) */
    backgroundValidationDanger: lightColorList.backgroundDanger,
    /** @deprecated Use `backgroundSuccess` (FEINF-xxxx) */
    backgroundValidationSuccess: lightColorList.backgroundSuccess,
    /** @deprecated Use `backgroundWarning` (FEINF-xxxx) */
    backgroundValidationWarning: lightColorList.backgroundWarning,
    /** @deprecated Use `border` (FEINF-xxxx) */
    borderDecorative: lightColorList.border,
    /** @deprecated Use `borderDanger` (FEINF-xxxx) */
    borderValidationDanger: lightColorList.borderDanger,
    /** @deprecated Use `borderWarning` (FEINF-xxxx) */
    borderValidationWarning: lightColorList.borderWarning,
    /** @deprecated Use `tableBackgroundUnselectedHover` (FEINF-xxxx) */
    tableRowHover: lightColorList.tableBackgroundUnselectedHover,
    /** @deprecated Use `textSecondary` (FEINF-xxxx) */
    textValidationInfo: lightColorList.textSecondary,
    /** @deprecated Use `codeBackground` (FEINF-xxxx) */
    typographyCodeBg: lightColorList.codeBackground,
    /** @deprecated Do not use Tag colors (go/updating-tag-colors). Ask in #dubois if you have any questions. */
    tagBrown: primitiveColors.brown,
    /** @deprecated Do not use Tag colors (go/updating-tag-colors). Ask in #dubois if you have any questions. */
    tagCharcoal: primitiveColors.grey600,
    /** @deprecated Do not use Tag colors (go/updating-tag-colors). Ask in #dubois if you have any questions. */
    tagCoral: primitiveColors.coral,
    /** @deprecated Do not use Tag colors (go/updating-tag-colors). Ask in #dubois if you have any questions. */
    tagDefault: primitiveColors.grey100,
    /** @deprecated Do not use Tag colors (go/updating-tag-colors). Ask in #dubois if you have any questions. */
    tagHover: 'rgba(34, 114, 180, 0.0800)',
    /** @deprecated Do not use Tag colors (go/updating-tag-colors). Ask in #dubois if you have any questions. */
    tagIconHover: primitiveColors.grey600,
    /** @deprecated Do not use Tag colors (go/updating-tag-colors). Ask in #dubois if you have any questions. */
    tagIconPress: primitiveColors.grey600,
    /** @deprecated Do not use Tag colors (go/updating-tag-colors). Ask in #dubois if you have any questions. */
    tagIndigo: primitiveColors.indigo,
    /** @deprecated Do not use Tag colors (go/updating-tag-colors). Ask in #dubois if you have any questions. */
    tagInverse: primitiveColors.grey800,
    /** @deprecated Do not use Tag colors (go/updating-tag-colors). Ask in #dubois if you have any questions. */
    tagLemon: primitiveColors.lemon,
    /** @deprecated Do not use Tag colors (go/updating-tag-colors). Ask in #dubois if you have any questions. */
    tagLime: primitiveColors.lime,
    /** @deprecated Do not use Tag colors (go/updating-tag-colors). Ask in #dubois if you have any questions. */
    tagPink: primitiveColors.pink,
    /** @deprecated Do not use Tag colors (go/updating-tag-colors). Ask in #dubois if you have any questions. */
    tagPress: 'rgba(34, 114, 180, 0.1600)',
    /** @deprecated Do not use Tag colors (go/updating-tag-colors). Ask in #dubois if you have any questions. */
    tagPurple: primitiveColors.purple,
    /** @deprecated Do not use Tag colors (go/updating-tag-colors). Ask in #dubois if you have any questions. */
    tagTeal: primitiveColors.teal,
    /** @deprecated Do not use Tag colors (go/updating-tag-colors). Ask in #dubois if you have any questions. */
    tagText: primitiveColors.white,
    /** @deprecated Do not use Tag colors (go/updating-tag-colors). Ask in #dubois if you have any questions. */
    tagTurquoise: primitiveColors.turquoise,
};
export const deprecatedSemanticColorsDark = {
    backgroundValidationDanger: darkColorList.backgroundDanger,
    backgroundValidationSuccess: darkColorList.backgroundSuccess,
    backgroundValidationWarning: darkColorList.backgroundWarning,
    borderDecorative: darkColorList.border,
    borderValidationDanger: darkColorList.borderDanger,
    borderValidationWarning: darkColorList.borderWarning,
    tableRowHover: darkColorList.tableBackgroundUnselectedHover,
    textValidationInfo: darkColorList.textSecondary,
    typographyCodeBg: darkColorList.codeBackground,
    tagBrown: 'rgba(166, 99, 12, 0.8600)',
    tagCharcoal: 'rgba(68, 83, 95, 0.8600)',
    tagCoral: 'rgba(200, 50, 67, 0.8600)',
    tagDefault: primitiveColors.grey650,
    tagHover: 'rgba(138, 202, 255, 0.0800)',
    tagIconHover: primitiveColors.grey350,
    tagIconPress: primitiveColors.grey350,
    tagIndigo: 'rgba(67, 74, 147, 0.8600)',
    tagInverse: primitiveColors.grey800,
    tagLemon: 'rgba(250, 203, 102, 0.8600)',
    tagLime: 'rgba(48, 134, 19, 0.8600)',
    tagPink: 'rgba(180, 80, 145, 0.8600)',
    tagPress: 'rgba(138, 202, 255, 0.1600)',
    tagPurple: 'rgba(138, 99, 191, 0.8600)',
    tagTeal: 'rgba(4, 134, 125, 0.8600)',
    tagText: primitiveColors.grey100,
    tagTurquoise: 'rgba(19, 125, 174, 0.8600)',
};
export function getColors(isDarkMode) {
    return {
        ...deprecatedPrimitiveColors,
        ...(isDarkMode ? deprecatedSemanticColorsDark : deprecatedSemanticColorsLight),
        ...getSemanticColors(isDarkMode),
    };
}
//# sourceMappingURL=colors.js.map