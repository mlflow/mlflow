/// <reference types="react" />
export interface FilterContextType {
    label: string;
    value: string[];
    isInsideFilter: boolean;
    multiSelect?: boolean;
    setValue: (value: string[]) => void;
    setIsControlled: (isControlled: boolean) => void;
}
export declare const FilterContext: import("react").Context<FilterContextType>;
export declare const FilterContextProvider: ({ children, value }: {
    children: JSX.Element;
    value: FilterContextType;
}) => import("@emotion/react/jsx-runtime").JSX.Element;
//# sourceMappingURL=FilterContext.d.ts.map