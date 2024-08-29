import { type ExtendedRefs } from '@floating-ui/react';
import type { UseComboboxReturnValue } from 'downshift';
import React from 'react';
import type { DesignSystemEventProviderAnalyticsEventTypes } from '../DesignSystemEventProvider';
import type { AnalyticsEventProps } from '../types';
export interface TypeaheadComboboxRootProps<T> extends React.HTMLAttributes<HTMLDivElement>, AnalyticsEventProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange> {
    comboboxState: UseComboboxReturnValue<T>;
    multiSelect?: boolean;
    floatingUiRefs?: ExtendedRefs<Element>;
    floatingStyles?: React.CSSProperties;
    children: React.ReactNode;
}
export declare const TypeaheadComboboxRoot: React.FC<TypeaheadComboboxRootProps<any>>;
export default TypeaheadComboboxRoot;
//# sourceMappingURL=TypeaheadComboboxRoot.d.ts.map