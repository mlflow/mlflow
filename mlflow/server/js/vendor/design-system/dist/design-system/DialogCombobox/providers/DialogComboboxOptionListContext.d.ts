export interface DialogComboboxOptionListContextType {
    isInsideDialogComboboxOptionList?: boolean;
    lookAhead: string;
    setLookAhead: (lookAhead: string) => void;
}
export declare const DialogComboboxOptionListContext: import("react").Context<DialogComboboxOptionListContextType>;
export declare const DialogComboboxOptionListContextProvider: ({ children, value, }: {
    children: JSX.Element;
    value: DialogComboboxOptionListContextType;
}) => import("@emotion/react/jsx-runtime").JSX.Element;
//# sourceMappingURL=DialogComboboxOptionListContext.d.ts.map