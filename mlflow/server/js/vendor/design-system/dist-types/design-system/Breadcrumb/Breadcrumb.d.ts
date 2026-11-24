import type { BreadcrumbProps as AntDBreadcrumbProps } from 'antd';
import { Breadcrumb as AntDBreadcrumb } from 'antd';
import type { DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
export interface BreadcrumbProps extends Pick<AntDBreadcrumbProps, 'itemRender' | 'params' | 'routes' | 'className'>, HTMLDataAttributes, DangerouslySetAntdProps<AntDBreadcrumbProps> {
    /** Include trailing caret */
    includeTrailingCaret?: boolean;
}
interface BreadcrumbInterface extends React.FC<BreadcrumbProps> {
    Item: typeof AntDBreadcrumb.Item;
    Separator: typeof AntDBreadcrumb.Separator;
}
export declare const Breadcrumb: BreadcrumbInterface;
export {};
//# sourceMappingURL=Breadcrumb.d.ts.map