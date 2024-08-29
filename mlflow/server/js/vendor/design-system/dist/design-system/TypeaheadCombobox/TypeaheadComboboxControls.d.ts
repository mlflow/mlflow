import type { UseComboboxGetToggleButtonPropsOptions } from 'downshift';
import { type DownshiftToggleButtonProps } from './TypeaheadComboboxToggleButton';
export interface TypeaheadComboboxControlsProps {
    getDownshiftToggleButtonProps: (options?: UseComboboxGetToggleButtonPropsOptions) => DownshiftToggleButtonProps;
    showClearSelectionButton?: boolean;
    showComboboxToggleButton?: boolean;
    handleClear?: (e: any) => void;
    disabled?: boolean;
}
export declare const TypeaheadComboboxControls: React.FC<any>;
//# sourceMappingURL=TypeaheadComboboxControls.d.ts.map