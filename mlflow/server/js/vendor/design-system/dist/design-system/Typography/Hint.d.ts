import { Typography as AntDTypography } from 'antd';
import type { ComponentProps } from 'react';
import type { DangerouslySetAntdProps, TypographyColor, HTMLDataAttributes } from '../types';
type AntDTypographyProps = ComponentProps<typeof AntDTypography>;
type AntDTextProps = ComponentProps<typeof AntDTypography['Text']>;
export interface TypographyHintProps extends AntDTypographyProps, Pick<AntDTextProps, 'ellipsis' | 'id' | 'title' | 'aria-label'>, HTMLDataAttributes, DangerouslySetAntdProps<Omit<AntDTextProps, 'hint'>> {
    bold?: boolean;
    size?: 'sm' | 'md' | 'lg' | 'xl' | 'xxl';
    withoutMargins?: boolean;
    color?: TypographyColor;
}
export declare function Hint(userProps: TypographyHintProps): JSX.Element;
export {};
//# sourceMappingURL=Hint.d.ts.map