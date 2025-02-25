export enum AvailableDesignSystemFlags {
  '__DEBUG__' = '__DEBUG__',
  'USE_FLEX_BUTTON' = 'USE_FLEX_BUTTON',
  'USE_NEW_CHECKBOX_STYLES' = 'USE_NEW_CHECKBOX_STYLES',
}

type DesignSystemFlagMetadata = {
  description: string;
  shortName: string;
};

export const AvailableDesignSystemFlagMetadata: Record<AvailableDesignSystemFlags, DesignSystemFlagMetadata> = {
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

export type DesignSystemFlags = Partial<Record<AvailableDesignSystemFlags, boolean>>;
