import type { ButtonProps as AntDButtonProps } from 'antd';
export interface MyNewComponentProps extends AntDButtonProps {
    /** These props all come from `antd`, and Ant doesn't annotate their
     * types (unfortunately). But, we can optionally decide to annotate them
     * on an individual basis. Pretty cool!
     */
    href?: AntDButtonProps['href'];
}
export declare const MyNewComponent: {
    (props: MyNewComponentProps): JSX.Element;
    defaultProps: Partial<MyNewComponentProps>;
};
//# sourceMappingURL=MyNewComponent.d.ts.map