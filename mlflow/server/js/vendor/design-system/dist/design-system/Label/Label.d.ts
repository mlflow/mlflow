import type { InfoPopoverProps } from '../Popover';
import type { HTMLDataAttributes } from '../types';
export interface LabelProps extends React.LabelHTMLAttributes<HTMLLabelElement>, HTMLDataAttributes {
    inline?: boolean;
    required?: boolean;
    /**
     * Adds an `InfoPopover` after the label. Please utilize `FormUI.Hint` unless a popover is absolutely necessary.
     * @type React.ReactNode | undefined
     */
    infoPopoverContents?: React.ReactNode;
    infoPopoverProps?: InfoPopoverProps;
}
export declare const Label: (props: LabelProps) => import("@emotion/react/jsx-runtime").JSX.Element;
//# sourceMappingURL=Label.d.ts.map