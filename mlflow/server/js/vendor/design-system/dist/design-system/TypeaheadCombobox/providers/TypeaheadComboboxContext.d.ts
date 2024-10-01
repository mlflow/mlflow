import type { ExtendedRefs } from '@floating-ui/react';
import type { DesignSystemEventProviderAnalyticsEventTypes } from '../../DesignSystemEventProvider/DesignSystemEventProvider';
import type { AnalyticsEventProps } from '../../types';
export interface TypeaheadComboboxContextType extends AnalyticsEventProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange> {
    isInsideTypeaheadCombobox: boolean;
    multiSelect?: boolean;
    floatingUiRefs?: ExtendedRefs<Element>;
    floatingStyles?: React.CSSProperties;
    inputWidth?: number;
    setInputWidth?: (width: number) => void;
}
export declare const TypeaheadComboboxContext: import("react").Context<TypeaheadComboboxContextType>;
export declare const TypeaheadComboboxContextProvider: ({ children, value, }: {
    children: JSX.Element;
    value: TypeaheadComboboxContextType;
}) => import("@emotion/react/jsx-runtime").JSX.Element;
//# sourceMappingURL=TypeaheadComboboxContext.d.ts.map