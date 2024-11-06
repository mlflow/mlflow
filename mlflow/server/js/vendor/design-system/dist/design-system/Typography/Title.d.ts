import { Typography as AntDTypography } from 'antd';
import type { ComponentProps } from 'react';
import type { DangerouslySetAntdProps, TypographyColor, HTMLDataAttributes } from '../types';
type AntDTypographyProps = ComponentProps<typeof AntDTypography>;
type AntDTitleProps = ComponentProps<typeof AntDTypography['Title']>;
export interface TypographyTitleProps extends AntDTypographyProps, Pick<AntDTitleProps, 'level' | 'ellipsis' | 'id' | 'title' | 'aria-label'>, HTMLDataAttributes, DangerouslySetAntdProps<AntDTitleProps> {
    withoutMargins?: boolean;
    color?: TypographyColor;
    /** Only controls the HTML element rendered, styles are controlled by `level` prop */
    elementLevel?: AntDTitleProps['level'];
}
export declare function Title(userProps: TypographyTitleProps): JSX.Element;
export {};
//# sourceMappingURL=Title.d.ts.map