import type { Theme, SerializedStyles, Interpolation } from '@emotion/react';
import type React from 'react';
export declare const getDialogComboboxOptionLabelWidth: (theme: Theme, width: number | string) => number | string;
export declare const getDialogComboboxOptionItemWrapperStyles: (theme: Theme) => SerializedStyles;
export declare const infoIconStyles: (theme: Theme) => Interpolation<Theme>;
export declare const getKeyboardNavigationFunctions: (handleSelect: (...args: any[]) => any, { onKeyDown, onMouseEnter, onDefaultKeyDown, }: {
    onKeyDown?: ((...args: any[]) => any) | undefined;
    onMouseEnter?: ((...args: any[]) => any) | undefined;
    onDefaultKeyDown?: ((...args: any[]) => any) | undefined;
}) => {
    onKeyDown: (e: React.KeyboardEvent<HTMLDivElement>) => void;
    onMouseEnter: (e: React.MouseEvent<HTMLDivElement>) => void;
};
export declare const resetTabIndexToFocusedElement: (elem: HTMLElement) => void;
export declare const dialogComboboxLookAheadKeyDown: (e: React.KeyboardEvent<any>, setLookAhead: (val: string) => void, lookAhead: string) => void;
//# sourceMappingURL=shared.d.ts.map