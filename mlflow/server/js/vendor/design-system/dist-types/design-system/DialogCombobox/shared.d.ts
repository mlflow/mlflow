import type React from 'react';
import type { Theme } from '../../theme';
export declare const getDialogComboboxOptionLabelWidth: (theme: Theme, width: string | number) => string | number;
export declare function isOptionDisabled(option: HTMLElement): boolean;
export declare function highlightFirstNonDisabledOption(firstOptionItem: Element, startAt?: 'start' | 'end', previousSelection?: HTMLElement): void;
export declare function findClosestOptionSibling(element: HTMLElement, direction: 'previous' | 'next'): HTMLElement | null;
export declare const highlightOption: (currentSelection: HTMLElement, prevSelection?: HTMLElement | undefined, focus?: boolean) => void;
export declare const findHighlightedOption: (options: HTMLElement[]) => HTMLElement | undefined;
export declare const getContentOptions: (element: HTMLElement) => HTMLElement[] | undefined;
export declare const getKeyboardNavigationFunctions: (handleSelect: (...args: any[]) => any, { onKeyDown, onMouseEnter, onDefaultKeyDown, disableMouseOver, setDisableMouseOver, }: {
    onKeyDown?: ((...args: any[]) => any) | undefined;
    onMouseEnter?: ((...args: any[]) => any) | undefined;
    onDefaultKeyDown?: ((...args: any[]) => any) | undefined;
    disableMouseOver: boolean;
    setDisableMouseOver: (disableMouseOver: boolean) => void;
}) => {
    onKeyDown: (e: React.KeyboardEvent<HTMLDivElement>) => void;
    onMouseMove: (e: React.MouseEvent<HTMLDivElement, MouseEvent>) => void;
    onMouseEnter: (e: React.MouseEvent<HTMLDivElement, MouseEvent>) => void;
};
export declare const resetTabIndexToFocusedElement: (elem: HTMLElement) => void;
export declare const dialogComboboxLookAheadKeyDown: (e: React.KeyboardEvent<any>, setLookAhead: (val: string) => void, lookAhead: string) => void;
//# sourceMappingURL=shared.d.ts.map