export interface SelectContextType {
    isSelect: boolean;
    placeholder?: string;
}
export declare const SelectContext: import("react").Context<SelectContextType>;
export declare const SelectContextProvider: ({ children, value }: {
    children: JSX.Element;
    value: SelectContextType;
}) => import("@emotion/react/jsx-runtime").JSX.Element;
//# sourceMappingURL=SelectContext.d.ts.map