import type React from 'react';
import type { Theme } from '../../theme';
export declare const getDialogComboboxOptionLabelWidth: (theme: Theme, width: number | string) => number | string;
export declare function isOptionDisabled(option: HTMLElement): boolean;
export declare function highlightFirstNonDisabledOption(firstOptionItem: Element, startAt?: 'start' | 'end', previousSelection?: HTMLElement): void;
export declare function findClosestOptionSibling(element: HTMLElement, direction: 'previous' | 'next'): HTMLElement | null;
export declare const highlightOption: (currentSelection: HTMLElement, prevSelection?: HTMLElement, focus?: boolean) => void;
export declare const findHighlightedOption: (options: HTMLElement[]) => HTMLElement | undefined;
export declare const getContentOptions: (element: HTMLElement) => HTMLElement[] | undefined;
export declare const getKeyboardNavigationFunctions: (handleSelect: (...args: any[]) => any, { onKeyDown, onMouseEnter, onDefaultKeyDown, }: {
    onKeyDown?: (...args: any[]) => any;
    onMouseEnter?: (...args: any[]) => any;
    onDefaultKeyDown?: (...args: any[]) => any;
}) => {
    onKeyDown: (e: React.KeyboardEvent<HTMLDivElement>) => void;
    onMouseEnter: (e: React.MouseEvent<HTMLDivElement>) => void;
};
export declare const resetTabIndexToFocusedElement: (elem: HTMLElement) => void;
export declare const dialogComboboxLookAheadKeyDown: (e: React.KeyboardEvent<any>, setLookAhead: (val: string) => void, lookAhead: string) => void;
//# sourceMappingURL=shared.d.ts.map