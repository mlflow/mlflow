import { type ExtendedRefs } from '@floating-ui/react';
import type { UseComboboxReturnValue } from 'downshift';
import React from 'react';
export interface TypeaheadComboboxRootProps<T> extends React.HTMLAttributes<HTMLDivElement> {
    comboboxState: UseComboboxReturnValue<T>;
    multiSelect?: boolean;
    floatingUiRefs?: ExtendedRefs<Element>;
    floatingStyles?: React.CSSProperties;
    children: React.ReactNode;
}
export declare const TypeaheadComboboxRoot: React.FC<TypeaheadComboboxRootProps<any>>;
export default TypeaheadComboboxRoot;
//# sourceMappingURL=TypeaheadComboboxRoot.d.ts.map