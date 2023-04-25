import * as Popover from '@radix-ui/react-popover';
import React from 'react';
import type { HTMLDataAttributes } from '../../design-system/types';
export interface FilterTriggerProps extends Popover.PopoverTriggerProps, HTMLDataAttributes {
    maxWidth?: number;
    removable?: boolean;
    onRemove?: () => void;
    withCountBadge?: boolean;
    allowClear?: boolean;
    onClear?: () => void;
    showBadgeAfterValueCount?: number;
    controlled?: boolean;
}
export declare const FilterTrigger: React.ForwardRefExoticComponent<FilterTriggerProps & React.RefAttributes<HTMLButtonElement>>;
//# sourceMappingURL=FilterTrigger.d.ts.map