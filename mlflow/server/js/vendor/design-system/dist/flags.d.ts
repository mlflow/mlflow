export declare enum AvailableDesignSystemFlags {
    '__DEBUG__' = "__DEBUG__",
    'USE_NEW_SELECT_DROPDOWN_STYLES' = "USE_NEW_SELECT_DROPDOWN_STYLES",
    'USE_FLEX_BUTTON' = "USE_FLEX_BUTTON",
    'USE_NEW_TREE' = "USE_NEW_TREE",
    'USE_TRANSPARENT_INPUT' = "USE_TRANSPARENT_INPUT",
    'USE_NEW_CHECKBOX_STYLES' = "USE_NEW_CHECKBOX_STYLES",
    'USE_UPDATED_TABLE_STYLES' = "USE_UPDATED_TABLE_STYLES"
}
type DesignSystemFlagMetadata = {
    description: string;
    shortName: string;
};
export declare const AvailableDesignSystemFlagMetadata: Record<AvailableDesignSystemFlags, DesignSystemFlagMetadata>;
export type DesignSystemFlags = Partial<Record<AvailableDesignSystemFlags, boolean>>;
export {};
//# sourceMappingURL=flags.d.ts.map