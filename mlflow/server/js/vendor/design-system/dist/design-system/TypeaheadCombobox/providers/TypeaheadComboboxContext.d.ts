/// <reference types="react" />
import type { ExtendedRefs } from '@floating-ui/react';
export interface TypeaheadComboboxContextType {
    isInsideTypeaheadCombobox: boolean;
    multiSelect?: boolean;
    floatingUiRefs?: ExtendedRefs<Element>;
    floatingStyles?: React.CSSProperties;
}
export declare const TypeaheadComboboxContext: import("react").Context<TypeaheadComboboxContextType>;
export declare const TypeaheadComboboxContextProvider: ({ children, value, }: {
    children: JSX.Element;
    value: TypeaheadComboboxContextType;
}) => import("@emotion/react/jsx-runtime").JSX.Element;
//# sourceMappingURL=TypeaheadComboboxContext.d.ts.map