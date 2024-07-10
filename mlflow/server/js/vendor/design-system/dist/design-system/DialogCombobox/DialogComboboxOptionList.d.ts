import React from 'react';
import type { WithLoadingState } from '../LoadingState/LoadingState';
import type { HTMLDataAttributes } from '../types';
export interface DialogComboboxOptionListProps extends HTMLDataAttributes, React.HTMLAttributes<HTMLDivElement>, WithLoadingState {
    children: any;
    loading?: boolean;
    withProgressiveLoading?: boolean;
}
export declare const DialogComboboxOptionList: React.ForwardRefExoticComponent<DialogComboboxOptionListProps & React.RefAttributes<HTMLDivElement>>;
//# sourceMappingURL=DialogComboboxOptionList.d.ts.map