import { memoize } from 'lodash';
import { primitiveColors } from './_generated/PrimitiveColors';
import { getBorders, getLegacyBorders } from './borders';
import responsive from './breakpoints';
import { getColors, getProtectedSemanticColors } from './colors';
import generalVariables, { getShadowVariables } from './generalVariables';
import { getGradients } from './gradients';
import { getShadows } from './shadows';
import spacing from './spacing';
import typography from './typography';
const defaultOptions = {
    enableAnimation: false,
    zIndexBase: 1000,
    useNewBorderColors: false,
};
// Function to get variables for a certain theme.
// End users should use useDesignSystemTheme instead.
export const getTheme = memoize(getThemeImpl, (isDarkMode, options = defaultOptions) => {
    return `${isDarkMode}-${options.enableAnimation}-${options.zIndexBase}-${options.useNewBorderColors}`;
});
function getThemeImpl(isDarkMode, options = defaultOptions) {
    return {
        colors: {
            ...getColors(isDarkMode),
            ...(options.useNewBorderColors && {
                border: isDarkMode ? primitiveColors.grey700 : primitiveColors.grey100,
            }),
        },
        gradients: getGradients(isDarkMode),
        spacing,
        general: {
            ...generalVariables,
            ...getShadowVariables(isDarkMode),
        },
        shadows: getShadows(isDarkMode),
        typography,
        legacyBorders: getLegacyBorders(),
        // TODO: Update to use `getBorders`
        borders: getBorders(),
        responsive,
        isDarkMode,
        options,
        DU_BOIS_INTERNAL_ONLY: {
            colors: getProtectedSemanticColors(isDarkMode),
        },
    };
}
//# sourceMappingURL=index.js.map