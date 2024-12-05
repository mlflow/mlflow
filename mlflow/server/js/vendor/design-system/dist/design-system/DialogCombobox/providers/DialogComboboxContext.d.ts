import type { DesignSystemEventProviderAnalyticsEventTypes } from '../../DesignSystemEventProvider/DesignSystemEventProvider';
import type { AnalyticsEventValueChangeNoPiiFlagProps } from '../../types';
export interface DialogComboboxContextType extends AnalyticsEventValueChangeNoPiiFlagProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange> {
    id?: string;
    label?: string | React.ReactNode;
    value: string[];
    isInsideDialogCombobox: boolean;
    multiSelect?: boolean;
    setValue: (value: string[]) => void;
    setIsControlled: (isControlled: boolean) => void;
    stayOpenOnSelection?: boolean;
    isOpen?: boolean;
    setIsOpen: (isOpen: boolean) => void;
    emptyText?: string;
    contentWidth: number | string | undefined;
    setContentWidth: (width: number | string | undefined) => void;
    textOverflowMode: 'ellipsis' | 'multiline';
    setTextOverflowMode: (mode: 'ellipsis' | 'multiline') => void;
    scrollToSelectedElement: boolean;
    rememberLastScrollPosition: boolean;
}
export declare const DialogComboboxContext: import("react").Context<DialogComboboxContextType>;
export declare const DialogComboboxContextProvider: ({ children, value, }: {
    children: JSX.Element;
    value: DialogComboboxContextType;
}) => import("@emotion/react/jsx-runtime").JSX.Element;
//# sourceMappingURL=DialogComboboxContext.d.ts.map