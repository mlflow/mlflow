import unitless from '@emotion/unitless';
import _, { memoize } from 'lodash';
import { shadowDarkRgb, shadowLightRgb } from '../../theme/generalVariables';
import { COMBOBOX_MENU_ITEM_PADDING } from '../_shared_';
import { ColorVars } from '../constants';
/**
 * Recursively appends `!important` to all CSS properties in an Emotion `CSSObject`.
 * Used to ensure that we always override Ant styles, without worrying about selector precedence.
 */
export function importantify(obj) {
    return _.mapValues(obj, (value, key) => {
        if (_.isString(value) || _.isNumber(value) || _.isBoolean(value)) {
            // Make sure we don't double-append important
            if (_.isString(value) && _.endsWith(value, '!important')) {
                return value;
            }
            if (_.isNumber(value)) {
                if (unitless[key]) {
                    return `${value}!important`;
                }
                return `${value}px!important`;
            }
            return `${value}!important`;
        }
        if (_.isNil(value)) {
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
export function getTypographyColor(theme, key, fallbackColor) {
    if (theme && key && Object(theme.colors).hasOwnProperty(ColorVars[key])) {
        return theme.colors[ColorVars[key]];
    }
    return fallbackColor ?? theme.colors.textPrimary;
}
/**
 * Returns validation color based on state, has default validation colors if params are not provided
 * @param theme
 * @param validationState
 * @param errorColor
 * @param warningColor
 * @param successColor
 */
export function getValidationStateColor(theme, validationState, { errorColor, warningColor, successColor, } = {}) {
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
export function getDarkModePortalStyles(theme, useNewShadows, useNewBorderColors) {
    if (!theme || !theme.isDarkMode) {
        return {};
    }
    return {
        border: `1px solid ${useNewBorderColors ? theme.colors.border : theme.colors.borderDecorative}`,
        // With the new shadows we have corrected the style of shadows for dark mode, and don't need to disable them.
        ...(useNewShadows ? {} : { boxShadow: 'none' }),
    };
}
const shadowCoverTop = (bgColor) => `linear-gradient(${bgColor} 30%, rgba(0, 0, 0, 0)) center top`;
const shadowCoverBot = (bgColor) => `linear-gradient(rgba(0, 0, 0, 0), ${bgColor} 70%) center bottom`;
const shadowCoverLeft = (bgColor) => `linear-gradient(to left, rgba(0, 0, 0, 0), ${bgColor} 30%) left center`;
const shadowCoverRight = (bgColor) => `linear-gradient(to left, ${bgColor} 70%, rgba(0, 0, 0, 0)) right center`;
const shadowTop = (shadowRgb) => `radial-gradient(
  farthest-side at 50% 0,
  rgba(${shadowRgb}, 0.2),
  rgba(${shadowRgb}, 0)
) center top`;
const shadowBot = (shadowRgb) => `radial-gradient(
  farthest-side at 50% 100%,
  rgba(${shadowRgb}, 0.2),
  rgba(${shadowRgb}, 0)
) center bottom`;
const shadowLeft = (shadowRgb) => `radial-gradient(
  farthest-side at 0 50%,
  rgba(${shadowRgb}, 0.2),
  rgba(${shadowRgb}, 0)
) left center`;
const shadowRight = (shadowRgb) => `radial-gradient(
  farthest-side at 100% 50%,
  rgba(${shadowRgb}, 0.2),
  rgba(${shadowRgb}, 0)
) right center`;
const shadowCoverBackgroundSizeVertical = '100% 40px';
const shadowCoverBackgroundSizeHorizontal = '40px 100%';
const shadowBackgroundSizeVertical = '100% 14px';
const shadowBackgroundSizeHorizontal = '14px 100%';
const getShadowScrollStylesMemoized = memoize(function getShadowScrollStylesMemoized(theme, backgroundColor, orientation = 'vertical') {
    const bgColor = backgroundColor ?? theme.colors.backgroundPrimary;
    const shadowColor = theme.isDarkMode ? shadowDarkRgb : shadowLightRgb;
    if (orientation === 'horizontal') {
        return {
            background: `
            ${shadowCoverLeft(bgColor)},
            ${shadowCoverRight(bgColor)},
            ${shadowLeft(shadowColor)},
            ${shadowRight(shadowColor)}`,
            backgroundRepeat: 'no-repeat',
            backgroundSize: `
            ${shadowCoverBackgroundSizeHorizontal},
            ${shadowCoverBackgroundSizeHorizontal},
            ${shadowBackgroundSizeHorizontal},
            ${shadowBackgroundSizeHorizontal}`,
            backgroundAttachment: 'local, local, scroll, scroll',
        };
    }
    return {
        background: `
          ${shadowCoverTop(bgColor)},
          ${shadowCoverBot(bgColor)},
          ${shadowTop(shadowColor)},
          ${shadowBot(shadowColor)}`,
        backgroundRepeat: 'no-repeat',
        backgroundSize: `
          ${shadowCoverBackgroundSizeVertical},
          ${shadowCoverBackgroundSizeVertical},
          ${shadowBackgroundSizeVertical},
          ${shadowBackgroundSizeVertical}`,
        backgroundAttachment: 'local, local, scroll, scroll',
    };
}, (theme, backgroundColor, orientation) => {
    return `${theme.isDarkMode}-${backgroundColor}-${orientation}`;
});
export const getShadowScrollStyles = function getShadowScrollStyles(theme, { backgroundColor, orientation } = {}) {
    return getShadowScrollStylesMemoized(theme, backgroundColor, orientation);
};
const getBottomOnlyShadowScrollStylesMemoized = memoize(function getBottomOnlyShadowScrollStylesMemoized(theme, backgroundColor) {
    const bgColor = backgroundColor ?? theme.colors.backgroundPrimary;
    return {
        background: `
          ${shadowCoverBot(bgColor)},
          ${shadowBot(theme.isDarkMode ? shadowDarkRgb : shadowLightRgb)}`,
        backgroundRepeat: 'no-repeat',
        backgroundSize: `
          ${shadowCoverBackgroundSizeVertical},
          ${shadowBackgroundSizeVertical}`,
        backgroundAttachment: 'local, scroll',
    };
});
export const getBottomOnlyShadowScrollStyles = function getBottomOnlyShadowScrollStyles(theme, { backgroundColor } = {}) {
    return getBottomOnlyShadowScrollStylesMemoized(theme, backgroundColor);
};
export function getVirtualizedComboboxMenuItemStyles(virtualItem) {
    const topBottomPadding = COMBOBOX_MENU_ITEM_PADDING[0] + COMBOBOX_MENU_ITEM_PADDING[2];
    const leftRightPadding = COMBOBOX_MENU_ITEM_PADDING[1] + COMBOBOX_MENU_ITEM_PADDING[3];
    return {
        position: 'absolute',
        width: `calc(100% - ${leftRightPadding}px)`,
        height: `${virtualItem.size - topBottomPadding}px`,
        transform: `translateY(${virtualItem.start}px)`,
    };
}
//# sourceMappingURL=css-utils.js.map