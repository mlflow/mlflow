import { Typography as AntDTypography } from 'antd';
import type { ComponentProps } from 'react';
import type { DangerouslySetAntdProps, TypographyColor, HTMLDataAttributes } from '../types';
type AntDTypographyProps = ComponentProps<typeof AntDTypography>;
type AntDTitleProps = ComponentProps<typeof AntDTypography['Title']>;
export interface TitleProps extends AntDTypographyProps, Pick<AntDTitleProps, 'level' | 'ellipsis' | 'id' | 'title' | 'aria-label'>, HTMLDataAttributes, DangerouslySetAntdProps<AntDTitleProps> {
    withoutMargins?: boolean;
    color?: TypographyColor;
}
export declare function Title(userProps: TitleProps): JSX.Element;
export {};
//# sourceMappingURL=Title.d.ts.map