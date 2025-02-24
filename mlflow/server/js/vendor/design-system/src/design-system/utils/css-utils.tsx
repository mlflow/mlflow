import type { CSSObject } from '@emotion/react';
import unitless from '@emotion/unitless';
import _, { memoize } from 'lodash';
import type { CSSProperties } from 'react';
import type { VirtualItem } from 'react-virtual';

import type { Theme } from '../../theme';
import { shadowDarkRgb, shadowLightRgb } from '../../theme/generalVariables';
import { COMBOBOX_MENU_ITEM_PADDING } from '../_shared_';
import { ColorVars } from '../constants';
import type { TypographyColor, ValidationState } from '../types';

/**
 * Recursively appends `!important` to all CSS properties in an Emotion `CSSObject`.
 * Used to ensure that we always override Ant styles, without worrying about selector precedence.
 */
export function importantify(obj: CSSObject): CSSObject {
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

    return importantify(value as CSSObject);
  }) as CSSObject;
}

/**
 * Returns a text color, in case of invalid/missing key and missing fallback color it will return textPrimary
 * @param theme
 * @param key - key of TypographyColor
 * @param fallbackColor - color to return as fallback -- used to remove tertiary check inline
 */
export function getTypographyColor(theme: Theme, key?: TypographyColor, fallbackColor?: string): string {
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
export function getValidationStateColor(
  theme: Theme,
  validationState?: ValidationState,
  {
    errorColor,
    warningColor,
    successColor,
  }: {
    errorColor?: string;
    warningColor?: string;
    successColor?: string;
  } = {},
): string | undefined {
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

export function getDarkModePortalStyles(theme: Theme, useNewShadows: boolean): React.CSSProperties {
  if (!theme || !theme.isDarkMode) {
    return {};
  }

  return {
    border: `1px solid ${theme.colors.borderDecorative}`,
    // With the new shadows we have corrected the style of shadows for dark mode, and don't need to disable them.
    ...(useNewShadows ? {} : { boxShadow: 'none' }),
  };
}

const shadowCoverTop = (bgColor: string) => `linear-gradient(${bgColor} 30%, rgba(0, 0, 0, 0)) center top`;
const shadowCoverBot = (bgColor: string) => `linear-gradient(rgba(0, 0, 0, 0), ${bgColor} 70%) center bottom`;
const shadowCoverLeft = (bgColor: string) => `linear-gradient(to left, rgba(0, 0, 0, 0), ${bgColor} 30%) left center`;
const shadowCoverRight = (bgColor: string) => `linear-gradient(to left, ${bgColor} 70%, rgba(0, 0, 0, 0)) right center`;
const shadowTop = (shadowRgb: string) => `radial-gradient(
  farthest-side at 50% 0,
  rgba(${shadowRgb}, 0.2),
  rgba(${shadowRgb}, 0)
) center top`;
const shadowBot = (shadowRgb: string) => `radial-gradient(
  farthest-side at 50% 100%,
  rgba(${shadowRgb}, 0.2),
  rgba(${shadowRgb}, 0)
) center bottom`;
const shadowLeft = (shadowRgb: string) => `radial-gradient(
  farthest-side at 0 50%,
  rgba(${shadowRgb}, 0.2),
  rgba(${shadowRgb}, 0)
) left center`;
const shadowRight = (shadowRgb: string) => `radial-gradient(
  farthest-side at 100% 50%,
  rgba(${shadowRgb}, 0.2),
  rgba(${shadowRgb}, 0)
) right center`;

const shadowCoverBackgroundSizeVertical = '100% 40px';
const shadowCoverBackgroundSizeHorizontal = '40px 100%';
const shadowBackgroundSizeVertical = '100% 14px';
const shadowBackgroundSizeHorizontal = '14px 100%';

type GetShadowScrollFunction = (
  theme: Theme,
  options?: { backgroundColor?: string; orientation?: 'vertical' | 'horizontal' },
) => Pick<CSSObject, 'background' | 'backgroundRepeat' | 'backgroundSize' | 'backgroundAttachment'>;

const getShadowScrollStylesMemoized = memoize(
  function getShadowScrollStylesMemoized(
    theme: Theme,
    backgroundColor?: string,
    orientation: 'vertical' | 'horizontal' = 'vertical',
  ) {
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
  },
  (theme, backgroundColor, orientation) => {
    return `${theme.isDarkMode}-${backgroundColor}-${orientation}`;
  },
);

export const getShadowScrollStyles: GetShadowScrollFunction = function getShadowScrollStyles(
  theme: Theme,
  { backgroundColor, orientation } = {},
) {
  return getShadowScrollStylesMemoized(theme, backgroundColor, orientation);
};

const getBottomOnlyShadowScrollStylesMemoized = memoize(function getBottomOnlyShadowScrollStylesMemoized(
  theme: Theme,
  backgroundColor?: string,
) {
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

export const getBottomOnlyShadowScrollStyles: GetShadowScrollFunction = function getBottomOnlyShadowScrollStyles(
  theme: Theme,
  { backgroundColor } = {},
) {
  return getBottomOnlyShadowScrollStylesMemoized(theme, backgroundColor);
};

export function getVirtualizedComboboxMenuItemStyles(virtualItem: VirtualItem): CSSProperties {
  const topBottomPadding = COMBOBOX_MENU_ITEM_PADDING[0] + COMBOBOX_MENU_ITEM_PADDING[2];
  const leftRightPadding = COMBOBOX_MENU_ITEM_PADDING[1] + COMBOBOX_MENU_ITEM_PADDING[3];
  return {
    position: 'absolute',
    width: `calc(100% - ${leftRightPadding}px)`,
    height: `${virtualItem.size - topBottomPadding}px`,
    transform: `translateY(${virtualItem.start}px)`,
  };
}
