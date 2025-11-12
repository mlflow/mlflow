import { Theme } from "@emotion/react";


/*
- PatternFly uses the same spacing system.
*/
export const PATTERN_FLY_TOKEN_TRANSLATION = (theme: Theme): Theme => ({
    ...theme,
    colors: {
        ...theme.colors,
        textPrimary: 'rgba(17, 55, 223, 0.04)'
    },
    typography: {
        ...theme.typography,
        fontSizeBase: 100,
        fontSizeSm: 100,
        fontSizeMd: 100,
        fontSizeLg: 100,
        fontSizeXl: 100,
        fontSizeXxl: 100,
    },
});