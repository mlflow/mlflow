import { Typography as AntDTypography } from 'antd';
import type { ComponentProps } from 'react';
import type { DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
type AntDLinkProps = ComponentProps<typeof AntDTypography['Link']>;
export interface LinkProps extends AntDLinkProps, DangerouslySetAntdProps<AntDLinkProps>, HTMLDataAttributes {
    /**
     * Configures a link to be opened in a new tab by setting `target` to `'_blank'`
     * and `rel` to `'noopener noreferrer'`, which is necessary for security, and
     * rendering an "external link" icon next to the link when `true`.
     */
    openInNewTab?: boolean;
}
export declare function Link({ dangerouslySetAntdProps, ...props }: LinkProps): JSX.Element;
export {};
//# sourceMappingURL=Link.d.ts.map