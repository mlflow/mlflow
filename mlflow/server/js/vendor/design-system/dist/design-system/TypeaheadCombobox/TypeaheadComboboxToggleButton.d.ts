import React from 'react';
export interface DownshiftToggleButtonProps {
    id: string;
    onClick: (e: React.SyntheticEvent) => void;
    tabIndex: number;
}
interface TypeaheadComboboxToggleButtonProps extends DownshiftToggleButtonProps {
    disabled?: boolean;
}
export declare const TypeaheadComboboxToggleButton: React.ForwardRefExoticComponent<TypeaheadComboboxToggleButtonProps & React.RefAttributes<HTMLButtonElement>>;
export {};
//# sourceMappingURL=TypeaheadComboboxToggleButton.d.ts.map