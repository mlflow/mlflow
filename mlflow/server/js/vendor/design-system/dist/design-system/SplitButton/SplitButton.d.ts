import type { DropdownMenuProps } from '@radix-ui/react-dropdown-menu';
import type { DropdownButtonProps } from './Dropdown/DropdownButton';
import type { DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
export interface SplitButtonMenuInfo {
    key: React.Key;
    keyPath: React.Key[];
    domEvent: React.SyntheticEvent<HTMLElement>;
}
export type SplitButtonProps = Omit<DropdownButtonProps, 'overlay' | 'type' | 'size' | 'trigger'> & HTMLDataAttributes & DangerouslySetAntdProps<Partial<DropdownButtonProps>> & {
    /**
     * @deprecated Please migrate to the DuBois DropdownMenu component and use the `menu` prop.
     */
    deprecatedMenu?: DropdownButtonProps['overlay'];
    /**
     * The visual style of the button, either default or primary
     */
    type?: 'default' | 'primary';
    loading?: boolean;
    loadingButtonStyles?: React.CSSProperties;
    /**
     * Props to be passed down to DropdownMenu.Root
     */
    dropdownMenuRootProps?: DropdownMenuProps;
};
export declare const SplitButton: React.FC<SplitButtonProps>;
//# sourceMappingURL=SplitButton.d.ts.map