/// <reference types="react" />
export interface DialogComboboxContextType {
    label: string | React.ReactNode;
    value: string[];
    isInsideDialogCombobox: boolean;
    multiSelect?: boolean;
    setValue: (value: string[]) => void;
    setIsControlled: (isControlled: boolean) => void;
    stayOpenOnSelection?: boolean;
    setIsOpen: (isOpen: boolean) => void;
}
export declare const DialogComboboxContext: import("react").Context<DialogComboboxContextType>;
export declare const DialogComboboxContextProvider: ({ children, value, }: {
    children: JSX.Element;
    value: DialogComboboxContextType;
}) => import("@emotion/react/jsx-runtime").JSX.Element;
//# sourceMappingURL=DialogComboboxContext.d.ts.map