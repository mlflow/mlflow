import { FormMessage } from '../FormMessage/FormMessage';
export * from './RHFAdapters';
export interface HorizontalFormSectionProps extends React.HTMLAttributes<HTMLDivElement> {
    children: React.ReactNode;
    labelColWidth?: number | string;
    inputColWidth?: number | string;
}
export declare const FormUI: {
    Message: typeof FormMessage;
    Label: (props: import("../Label/Label").LabelProps) => import("@emotion/react/jsx-runtime").JSX.Element;
    Hint: (props: React.HTMLAttributes<HTMLSpanElement>) => import("@emotion/react/jsx-runtime").JSX.Element;
    HorizontalFormRow: ({ children, labelColWidth, inputColWidth, ...restProps }: HorizontalFormSectionProps) => import("@emotion/react/jsx-runtime").JSX.Element;
};
//# sourceMappingURL=index.d.ts.map