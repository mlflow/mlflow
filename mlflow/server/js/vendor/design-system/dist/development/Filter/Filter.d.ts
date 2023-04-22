import * as Popover from '@radix-ui/react-popover';
import type { HTMLDataAttributes } from '../../design-system/types';
export interface FilterRootProps extends Popover.PopoverProps, HTMLDataAttributes {
    label: string;
    value?: string[];
    stayOpenOnSelection?: boolean;
    multiSelect?: boolean;
}
export declare const Filter: ({ children, label, value, ...props }: FilterRootProps) => import("@emotion/react/jsx-runtime").JSX.Element;
//# sourceMappingURL=Filter.d.ts.map