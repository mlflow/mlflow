declare const antdGeneralVariables: {
    classnamePrefix: string;
    iconfontCssPrefix: string;
    borderRadiusBase: number;
    borderWidth: number;
    heightSm: number;
    heightBase: number;
    iconSize: number;
    iconFontSize: number;
    buttonHeight: number;
    buttonInnerHeight: number;
};
export declare const shadowLightRgb: "31, 39, 45";
export declare const shadowDarkRgb: "0, 0, 0";
export declare const getShadowVariables: (isDarkMode: boolean) => {
    /**
     * @deprecated Use new shadow variables under theme.shadows.*, check out go/dubois-elevation-doc decision doc for guidance on choosing the new shadow variable
     */
    readonly shadowLow: "0px 4px 16px rgba(0, 0, 0, 0.12)";
    /**
     * @deprecated Use new shadow variables under theme.shadows.*, check out go/dubois-elevation-doc decision doc for guidance on choosing the new shadow variable
     */
    readonly shadowHigh: "0px 8px 24px rgba(0, 0, 0, 0.2);";
} | {
    /**
     * @deprecated Use new shadow variables under theme.shadows.*, check out go/dubois-elevation-doc decision doc for guidance on choosing the new shadow variable
     */
    readonly shadowLow: "0px 4px 16px rgba(31, 39, 45, 0.12)";
    /**
     * @deprecated Use new shadow variables under theme.shadows.*, check out go/dubois-elevation-doc decision doc for guidance on choosing the new shadow variable
     */
    readonly shadowHigh: "0px 8px 24px rgba(31, 39, 45, 0.2)";
};
export default antdGeneralVariables;
//# sourceMappingURL=generalVariables.d.ts.map