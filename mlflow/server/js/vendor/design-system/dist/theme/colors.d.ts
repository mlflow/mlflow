import { primitiveColors } from './_generated/PrimitiveColors';
import { lightColorList } from './_generated/SemanticColors-Light';
declare const branded: {
    ai: {
        /** For AI components, the top-left-oriented start color of gradient treatments. */
        gradientStart: string;
        /** For AI components, the mid color of gradient treatments. */
        gradientMid: string;
        /** For AI components, the bottom-right-oriented end color of gradient treatments. */
        gradientEnd: string;
    };
};
type DesignSystemColors = typeof lightColorList & typeof primitiveColors & {
    /** These colors represent specific brand interactions and experiences,
     * and should be only be used in specific contexts.
     */
    branded: typeof branded;
};
export declare function getProtectedSemanticColors(isDarkMode: boolean): {
    tagBackgroundBrown: string;
    tagBackgroundCharcoal: string;
    tagBackgroundCoral: string;
    tagBackgroundDefault: string;
    tagBackgroundIndigo: string;
    tagBackgroundLemon: string;
    tagBackgroundLime: string;
    tagBackgroundPink: string;
    tagBackgroundPurple: string;
    tagBackgroundTeal: string;
    tagBackgroundTurquoise: string;
    tagIconBrown: string;
    tagIconCharcoal: string;
    tagIconCoral: string;
    tagIconDefault: string;
    tagIconIndigo: string;
    tagIconLemon: string;
    tagIconLime: string;
    tagIconPink: string;
    tagIconPurple: string;
    tagIconTeal: string;
    tagIconTurquoise: string;
    tagTextBrown: string;
    tagTextCharcoal: string;
    tagTextCoral: string;
    tagTextDefault: string;
    tagTextIndigo: string;
    tagTextLemon: string;
    tagTextLime: string;
    tagTextPink: string;
    tagTextPurple: string;
    tagTextTeal: string;
    tagTextTurquoise: string;
};
export declare function getAntdColors(isDarkMode: boolean): {
    alertTextColor: string;
    alertMessageColor: string;
    spinDotDefault: string;
    alertErrorBgColor: string;
    alertInfoBgColor: string;
    alertWarningBgColor: string;
    bodyBackground: string;
    textColor: string;
    textColorSecondary: string;
    primaryColor: string;
    infoColor: string;
    errorColor: string;
    successColor: string;
    warningColor: string;
    borderColorBase: string;
    alertErrorTextColor: string;
    alertErrorBorderColor: string;
    alertErrorIconColor: string;
    alertInfoTextColor: string;
    alertInfoBorderColor: string;
    alertInfoIconColor: string;
    alertWarningTextColor: string;
    alertWarningBorderColor: string;
    alertWarningIconColor: string;
};
declare const deprecatedPrimitiveColors: {
    /** @deprecated This was an alias to `primitiveColors.blue600`, please use that instead.
     * If possible, please use an appropriate semantic color, such as `actionPrimaryBackgroundDefault`. */
    primary: string;
    /** @deprecated This was an alias to `primitiveColors.grey600`, please use that instead.
     * If possible, please use an appropriate semantic color, such as `actionPrimaryBackgroundHover`. */
    charcoal: string;
    /** @deprecated This color supports legacy radio styles and will be removed in an upcoming release (FEINF-1674). */
    radioInteractiveAvailable: string;
    /** @deprecated This color supports legacy radio styles and will be removed in an upcoming release (FEINF-1674). */
    radioInteractiveHover: string;
    /** @deprecated This color supports legacy radio styles and will be removed in an upcoming release (FEINF-1674). */
    radioInteractivePress: string;
    /** @deprecated This color supports legacy radio styles and will be removed in an upcoming release (FEINF-1674). */
    radioDisabled: string;
    /** @deprecated This color supports legacy radio styles and will be removed in an upcoming release (FEINF-1674). */
    radioDefaultBorder: string;
    /** @deprecated This color supports legacy radio styles and will be removed in an upcoming release (FEINF-1674). */
    radioDefaultBackground: string;
    /** @deprecated This color supports legacy radio styles and will be removed in an upcoming release (FEINF-1674). */
    radioInteractiveHoverSecondary: string;
    /** @deprecated This color supports legacy radio styles and will be removed in an upcoming release (FEINF-1674). */
    radioInteractivePressSecondary: string;
};
export declare const deprecatedSemanticColorsLight: {
    /** @deprecated Use `backgroundDanger` (FEINF-xxxx) */
    backgroundValidationDanger: string;
    /** @deprecated Use `backgroundSuccess` (FEINF-xxxx) */
    backgroundValidationSuccess: string;
    /** @deprecated Use `backgroundWarning` (FEINF-xxxx) */
    backgroundValidationWarning: string;
    /** @deprecated Use `border` (FEINF-xxxx) */
    borderDecorative: string;
    /** @deprecated Use `borderDanger` (FEINF-xxxx) */
    borderValidationDanger: string;
    /** @deprecated Use `borderWarning` (FEINF-xxxx) */
    borderValidationWarning: string;
    /** @deprecated Use `tableBackgroundUnselectedHover` (FEINF-xxxx) */
    tableRowHover: string;
    /** @deprecated Use `textSecondary` (FEINF-xxxx) */
    textValidationInfo: string;
    /** @deprecated Use `codeBackground` (FEINF-xxxx) */
    typographyCodeBg: string;
    /** @deprecated Do not use Tag colors (go/updating-tag-colors). Ask in #dubois if you have any questions. */
    tagBrown: string;
    /** @deprecated Do not use Tag colors (go/updating-tag-colors). Ask in #dubois if you have any questions. */
    tagCharcoal: string;
    /** @deprecated Do not use Tag colors (go/updating-tag-colors). Ask in #dubois if you have any questions. */
    tagCoral: string;
    /** @deprecated Do not use Tag colors (go/updating-tag-colors). Ask in #dubois if you have any questions. */
    tagDefault: string;
    /** @deprecated Do not use Tag colors (go/updating-tag-colors). Ask in #dubois if you have any questions. */
    tagHover: string;
    /** @deprecated Do not use Tag colors (go/updating-tag-colors). Ask in #dubois if you have any questions. */
    tagIconHover: string;
    /** @deprecated Do not use Tag colors (go/updating-tag-colors). Ask in #dubois if you have any questions. */
    tagIconPress: string;
    /** @deprecated Do not use Tag colors (go/updating-tag-colors). Ask in #dubois if you have any questions. */
    tagIndigo: string;
    /** @deprecated Do not use Tag colors (go/updating-tag-colors). Ask in #dubois if you have any questions. */
    tagInverse: string;
    /** @deprecated Do not use Tag colors (go/updating-tag-colors). Ask in #dubois if you have any questions. */
    tagLemon: string;
    /** @deprecated Do not use Tag colors (go/updating-tag-colors). Ask in #dubois if you have any questions. */
    tagLime: string;
    /** @deprecated Do not use Tag colors (go/updating-tag-colors). Ask in #dubois if you have any questions. */
    tagPink: string;
    /** @deprecated Do not use Tag colors (go/updating-tag-colors). Ask in #dubois if you have any questions. */
    tagPress: string;
    /** @deprecated Do not use Tag colors (go/updating-tag-colors). Ask in #dubois if you have any questions. */
    tagPurple: string;
    /** @deprecated Do not use Tag colors (go/updating-tag-colors). Ask in #dubois if you have any questions. */
    tagTeal: string;
    /** @deprecated Do not use Tag colors (go/updating-tag-colors). Ask in #dubois if you have any questions. */
    tagText: string;
    /** @deprecated Do not use Tag colors (go/updating-tag-colors). Ask in #dubois if you have any questions. */
    tagTurquoise: string;
};
export declare const deprecatedSemanticColorsDark: typeof deprecatedSemanticColorsLight;
export type SecondaryColorToken = 'brown' | 'coral' | 'indigo' | 'lemon' | 'lime' | 'pink' | 'purple' | 'teal' | 'turquoise';
export type TagColorToken = `tag${Capitalize<SecondaryColorToken>}` | 'tagDefault';
export declare function getColors(isDarkMode: boolean): typeof deprecatedPrimitiveColors & typeof deprecatedSemanticColorsLight & DesignSystemColors;
export {};
//# sourceMappingURL=colors.d.ts.map