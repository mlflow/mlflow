import spacing from './spacing';
import antdVars from '../antd-vars';
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
    buttonInnerHeight: heightBase - spacing.sm * 2 - borderWidth * 2,
};
export const shadowLightRgb = '31, 39, 45';
export const shadowDarkRgb = '0, 0, 0';
export const getShadowVariables = (isDarkMode) => {
    if (isDarkMode) {
        return {
            /**
             * @deprecated Use new shadow variables under theme.shadows.*, check out go/dubois-elevation-doc decision doc for guidance on choosing the new shadow variable
             */
            shadowLow: `0px 4px 16px rgba(${shadowDarkRgb}, 0.12)`,
            /**
             * @deprecated Use new shadow variables under theme.shadows.*, check out go/dubois-elevation-doc decision doc for guidance on choosing the new shadow variable
             */
            shadowHigh: `0px 8px 24px rgba(${shadowDarkRgb}, 0.2);`,
        };
    }
    else {
        return {
            /**
             * @deprecated Use new shadow variables under theme.shadows.*, check out go/dubois-elevation-doc decision doc for guidance on choosing the new shadow variable
             */
            shadowLow: `0px 4px 16px rgba(${shadowLightRgb}, 0.12)`,
            /**
             * @deprecated Use new shadow variables under theme.shadows.*, check out go/dubois-elevation-doc decision doc for guidance on choosing the new shadow variable
             */
            shadowHigh: `0px 8px 24px rgba(${shadowLightRgb}, 0.2)`,
        };
    }
};
export default antdGeneralVariables;
//# sourceMappingURL=generalVariables.js.map