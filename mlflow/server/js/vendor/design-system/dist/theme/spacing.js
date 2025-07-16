export const DEFAULT_SPACING_UNIT = 8;
export const MODAL_PADDING = 40;
const spacing = {
    xs: DEFAULT_SPACING_UNIT / 2,
    sm: DEFAULT_SPACING_UNIT,
    md: DEFAULT_SPACING_UNIT * 2,
    lg: DEFAULT_SPACING_UNIT * 3,
};
// Less variables that are used by AntD
export const antdSpacing = {
    defaultPaddingLg: spacing.lg,
    defaultPaddingMd: spacing.md,
    defaultPaddingSm: spacing.sm,
    defaultPaddingXs: spacing.sm,
    defaultPaddingXss: spacing.xs,
    paddingLg: spacing.md,
    // TODO: Check if there is a better alternative with team
    paddingMd: spacing.sm,
    paddingSm: spacing.sm,
    paddingXs: spacing.xs,
    paddingXss: 0,
    marginSm: 12,
    marginLg: spacing.lg,
    // Button
    btnPaddingHorizontalBase: DEFAULT_SPACING_UNIT * 2,
    btnPaddingHorizontalLg: DEFAULT_SPACING_UNIT * 2,
    btnPaddingHorizontalSm: DEFAULT_SPACING_UNIT * 2,
    // Input
    inputPaddingHorizontal: DEFAULT_SPACING_UNIT * 1.5,
    inputPaddingHorizontalBase: DEFAULT_SPACING_UNIT * 1.5,
    inputPaddingHorizontalSm: DEFAULT_SPACING_UNIT * 1.5,
    inputPaddingHorizontalLg: DEFAULT_SPACING_UNIT * 1.5,
    inputPaddingVertical: spacing.xs + 1,
    inputPaddingVerticalBase: spacing.xs + 1,
    inputPaddingVerticalSm: spacing.xs + 1,
    inputPaddingVerticalLg: spacing.xs + 1,
    // Modal
    modalPadding: MODAL_PADDING,
    modalLessPadding: MODAL_PADDING - 20,
    modalHeaderPadding: `${MODAL_PADDING}px ${MODAL_PADDING}px ${MODAL_PADDING - 20}px`,
    modalHeaderCloseSize: MODAL_PADDING * 2 + 22,
    modalHeaderBorderWidth: 0,
    modalBodyPadding: `0 ${MODAL_PADDING}px`,
    modalFooterPaddingVertical: 0,
    modalFooterPaddingHorizontal: 0,
    modalFooterBorderWidth: 0,
    // Switch
    switchPadding: 0,
    switchHeight: 16,
    switchMinWidth: 28,
    switchPinSize: 14,
};
export default spacing;
//# sourceMappingURL=spacing.js.map