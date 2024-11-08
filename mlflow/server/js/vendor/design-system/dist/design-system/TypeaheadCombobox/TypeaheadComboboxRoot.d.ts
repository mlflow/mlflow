import { type ExtendedRefs } from '@floating-ui/react';
import React from 'react';
import type { ComboboxStateAnalyticsReturnValue } from './hooks';
export interface TypeaheadComboboxRootProps<T> extends React.HTMLAttributes<HTMLDivElement> {
    comboboxState: ComboboxStateAnalyticsReturnValue<T>;
    multiSelect?: boolean;
    floatingUiRefs?: ExtendedRefs<Element>;
    floatingStyles?: React.CSSProperties;
    children: React.ReactNode;
}
export declare const TypeaheadComboboxRoot: React.FC<TypeaheadComboboxRootProps<any>>;
export default TypeaheadComboboxRoot;
//# sourceMappingURL=TypeaheadComboboxRoot.d.ts.map