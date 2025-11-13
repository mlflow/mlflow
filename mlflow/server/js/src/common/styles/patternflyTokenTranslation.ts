import { Theme } from "@emotion/react";
import { darkColorList, lightColorList } from "./patternflyStyles/patternflyColors";
import { patternflySpacing } from "./patternflyStyles/patternflySpacing";
import { getPatternflyShadowVariables, patternflyGeneral } from "./patternflyStyles/patternflyGeneral";
import { patternflyTypography } from "./patternflyStyles/patternflyTypography";
import { getPatternflyShadows } from "./patternflyStyles/patternflyShadows";
import { patternflyBorders, patternflyLegacyBorders } from "./patternflyStyles/patternflyBorders";
import { patternflyResponsive } from "./patternflyStyles/patternflyResponsive";

export const PATTERN_FLY_TOKEN_TRANSLATION = (theme: Theme, isDarkTheme: boolean): Theme => ({
    ...theme,
    colors: {
        ...theme.colors,
        ...(isDarkTheme ? darkColorList : lightColorList),
    },
    spacing: {
        ...theme.spacing,
        ...patternflySpacing,
    },
    general: {
        ...theme.general,
        ...patternflyGeneral,
        ...getPatternflyShadowVariables(isDarkTheme),
    },
    typography: {
        ...theme.typography,
        ...patternflyTypography,
    },
    shadows: {
        ...theme.shadows,
        ...getPatternflyShadows(isDarkTheme),
    },
    borders: {
        ...theme.borders,
        ...patternflyBorders,
    },
    legacyBorders: {
        ...theme.legacyBorders,
        ...patternflyLegacyBorders,
    },
    responsive: {
        ...theme.responsive,
        ...patternflyResponsive,
    },
    // Note: gradients skipped - PatternFly has very limited gradient tokens
    // Note: DU_BOIS_INTERNAL_ONLY skipped - internal use only, no PatternFly equivalent
});