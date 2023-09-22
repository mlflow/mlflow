/// <reference types="react" />
import type { UseComboboxReturnValue } from 'downshift';
import type { Theme } from '../../theme';
export interface TypeaheadComboboxMenuItemProps<T> extends React.HTMLAttributes<HTMLElement> {
    item: T;
    index: number;
    comboboxState: UseComboboxReturnValue<T>;
    textOverflowMode?: 'ellipsis' | 'multiline';
    isDisabled?: boolean;
    disabledReason?: React.ReactNode;
    hintContent?: React.ReactNode;
    className?: string;
    children?: React.ReactNode;
    _TYPE?: string;
}
export declare const getMenuItemStyles: (theme: Theme, isHighlighted: boolean, disabled?: boolean) => import("@emotion/utils").SerializedStyles;
export declare const TypeaheadComboboxMenuItem: import("react").ForwardRefExoticComponent<TypeaheadComboboxMenuItemProps<any> & import("react").RefAttributes<HTMLLIElement>>;
export default TypeaheadComboboxMenuItem;
//# sourceMappingURL=TypeaheadComboboxMenuItem.d.ts.map