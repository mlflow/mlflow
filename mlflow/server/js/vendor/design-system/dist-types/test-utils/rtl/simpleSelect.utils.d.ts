export declare const simpleSelectTestUtils: {
    getSelectedOptionLabelFromTrigger: (name?: string | RegExp | undefined) => string | null;
    getSelectedOptionValueFromTrigger: (name?: string | RegExp | undefined) => string | null;
    getSelectedOptionFromTrigger: (name?: string | RegExp | undefined) => {
        label: string | null;
        value: string | null;
    };
    expectSelectedOptionFromTriggerToBe: (label: string | RegExp, name?: string | RegExp | undefined) => void;
    toggleSelect: (name?: string | RegExp | undefined) => void;
    expectSelectToBeOpen: () => void;
    expectSelectToBeClosed: () => void;
    getOptionsLength: () => number;
    getAllOptions: () => (string | null)[];
    expectOptionsLengthToBe: (length: number) => void;
    getUnselectedOption: (label: string | RegExp) => HTMLElement;
    getSelectedOption: (label: string | RegExp) => HTMLElement;
    getOption: (label: string | RegExp) => HTMLElement;
    selectOption: (label: string | RegExp) => void;
    expectSelectedOptionToBe: (label: string | RegExp) => void;
};
//# sourceMappingURL=simpleSelect.utils.d.ts.map