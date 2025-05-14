import React from 'react';
import type { TypographyTitleProps } from '../Typography/Title';
import type { DangerousGeneralProps, HTMLDataAttributes } from '../types';
export interface HeaderProps extends HTMLDataAttributes, DangerousGeneralProps {
    /** The title for this page */
    title: React.ReactNode;
    /** Inline elements to be appended to the end of the title, such as a `Tag` */
    titleAddOns?: React.ReactNode | React.ReactNode[];
    /** A single `<Breadcrumb />` component */
    breadcrumbs?: React.ReactNode;
    /** An array of Dubois `<Button />` components */
    buttons?: React.ReactNode | React.ReactNode[];
    /** HTML title element level. This only controls the element rendered, title will look like a h2 */
    titleElementLevel?: TypographyTitleProps['elementLevel'];
}
export declare const Header: React.FC<HeaderProps>;
//# sourceMappingURL=Header.d.ts.map