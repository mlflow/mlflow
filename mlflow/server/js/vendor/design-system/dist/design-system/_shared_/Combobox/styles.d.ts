import type { SerializedStyles } from '@emotion/react';
import type { Theme } from '../../../theme';
import type { DialogComboboxContextType } from '../../DialogCombobox/providers/DialogComboboxContext';
export declare const getComboboxContentWrapperStyles: (theme: Theme, { maxHeight, maxWidth, minHeight, minWidth, width, useNewShadows, }: {
    maxHeight?: number | string;
    maxWidth?: number | string;
    minHeight?: number | string;
    minWidth?: number | string;
    width?: number | string;
    useNewShadows: boolean;
}) => SerializedStyles;
export declare const COMBOBOX_MENU_ITEM_PADDING: number[];
export declare const getComboboxOptionItemWrapperStyles: (theme: Theme) => SerializedStyles;
interface getComboboxOptionLabelStylesProps {
    theme: Theme;
    dangerouslyHideCheck?: boolean;
    textOverflowMode?: 'ellipsis' | 'multiline';
    contentWidth?: DialogComboboxContextType['contentWidth'];
    hasHintColumn?: boolean;
}
export declare const getComboboxOptionLabelStyles: ({ theme, dangerouslyHideCheck, textOverflowMode, contentWidth, hasHintColumn, }: getComboboxOptionLabelStylesProps) => SerializedStyles;
export declare const getInfoIconStyles: (theme: Theme) => SerializedStyles;
export declare const getCheckboxStyles: (theme: Theme, textOverflowMode: "ellipsis" | "multiline") => SerializedStyles;
export declare const getFooterStyles: (theme: Theme) => SerializedStyles;
export declare const getSelectItemWithHintColumnStyles: (hintColumnWidthPercent?: number) => SerializedStyles;
export declare const getHintColumnStyles: (theme: Theme, disabled: boolean, textOverflowMode: DialogComboboxContextType["textOverflowMode"]) => SerializedStyles;
export {};
//# sourceMappingURL=styles.d.ts.map