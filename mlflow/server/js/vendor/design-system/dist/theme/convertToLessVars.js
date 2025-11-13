// @ts-expect-error antd/dist/theme isn't typed
import { getThemeVariables } from 'antd/dist/theme';
import _ from 'lodash';
import { getAntdColors } from './colors';
import generalVariables from './generalVariables';
import { antdSpacing } from './spacing';
import typography from './typography';
/**
 * Condense our variables into a single object that will be used
 * to build exported `less` configurations.
 */
export function getLessVariables(isDarkMode) {
    return {
        ...(isDarkMode ? getThemeVariables({ dark: true }) : {}),
        // We store our tokens as objects with camelCase keys for JS compatibility. Here we convert them to kebab-case for
        // LESS compatibility.
        ..._.mapKeys(getAntdColors(isDarkMode), (_value, key) => _.kebabCase(key)),
        ..._.mapKeys(antdSpacing, (_value, key) => _.kebabCase(key)),
        ..._.mapKeys(typography, (_value, key) => _.kebabCase(key)),
        ..._.mapKeys(generalVariables, (_value, key) => _.kebabCase(key)),
    };
}
/**
 * Use this to modify variables before passing them on to `less`; Useful
 * if you want to deal with them in source as a different format,
 * i.e. for better TypeScript types.
 */
const transformLessVariableValue = (lessVariable) => {
    // Integers are converted to pixels. Use strings for other CSS dimensions.
    if (typeof lessVariable === 'number' && Number.isInteger(lessVariable)) {
        lessVariable = `${lessVariable}px`;
    }
    return lessVariable;
};
/**
 * This is used if you want to pass the theme to `modifyVars`.
 * We use that in `rollup`, but not in `storybook` since we want theme changes
 * to rebuild and refresh in the latter.
 */
export function getLessVariablesObject(isDarkMode) {
    const lessVariables = getLessVariables(isDarkMode);
    const lessVariablesObject = {};
    Object.entries(lessVariables).forEach(([key, value]) => {
        lessVariablesObject[`@${key}`] = transformLessVariableValue(value);
    });
    return lessVariablesObject;
}
/**
 * This exports a synthetic `less` file containing our `less` variables.
 * We use this in storybook so that we can force that environment to refresh
 * as we work on the theme during active development.
 */
export function getLessVariablesText(isDarkMode) {
    const lessVariables = getLessVariables(isDarkMode);
    return Object.keys(lessVariables)
        .map((key) => {
        return `@${key}: ${transformLessVariableValue(lessVariables[key])};`;
    })
        .join('\n');
}
//# sourceMappingURL=convertToLessVars.js.map