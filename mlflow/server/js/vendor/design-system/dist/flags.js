export var AvailableDesignSystemFlags;
(function (AvailableDesignSystemFlags) {
    AvailableDesignSystemFlags["__DEBUG__"] = "__DEBUG__";
    AvailableDesignSystemFlags["USE_FLEX_BUTTON"] = "USE_FLEX_BUTTON";
    AvailableDesignSystemFlags["USE_NEW_CHECKBOX_STYLES"] = "USE_NEW_CHECKBOX_STYLES";
})(AvailableDesignSystemFlags || (AvailableDesignSystemFlags = {}));
export const AvailableDesignSystemFlagMetadata = {
    [AvailableDesignSystemFlags.__DEBUG__]: {
        description: 'This flag is only used for testing. Do not use.',
        shortName: 'debug_do_not_use',
    },
    [AvailableDesignSystemFlags.USE_FLEX_BUTTON]: {
        description: 'When enabled, this flag will render Buttons with flexbox styles',
        shortName: 'Use flex Button',
    },
    [AvailableDesignSystemFlags.USE_NEW_CHECKBOX_STYLES]: {
        description: 'When enabled, this flag will render Checkboxes with the new design',
        shortName: 'Use new checkbox styles',
    },
};
//# sourceMappingURL=flags.js.map