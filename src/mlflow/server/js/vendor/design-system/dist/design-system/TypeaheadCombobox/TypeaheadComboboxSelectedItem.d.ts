/// <reference types="react" />
import { type SerializedStyles } from '@emotion/react';
import type { UseMultipleSelectionGetSelectedItemPropsOptions } from 'downshift';
import type { Theme } from '../../theme';
export interface TypeaheadComboboxSelectedItemProps<T> {
    label: React.ReactNode;
    item: any;
    getSelectedItemProps: (options: UseMultipleSelectionGetSelectedItemPropsOptions<T>) => any;
    removeSelectedItem: (item: T) => void;
}
export declare const getSelectedItemStyles: (theme: Theme) => SerializedStyles;
export declare const TypeaheadComboboxSelectedItem: React.FC<any>;
//# sourceMappingURL=TypeaheadComboboxSelectedItem.d.ts.map