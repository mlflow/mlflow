import { Typography as AntDTypography } from 'antd';
import type { ComponentProps } from 'react';
import type { DangerouslySetAntdProps, TypographyColor, HTMLDataAttributes } from '../types';
type AntDTypographyProps = ComponentProps<typeof AntDTypography>;
type AntDParagraphProps = ComponentProps<typeof AntDTypography['Paragraph']>;
export interface TypographyParagraphProps extends AntDTypographyProps, Pick<AntDParagraphProps, 'ellipsis' | 'disabled' | 'id' | 'title' | 'aria-label'>, HTMLDataAttributes, DangerouslySetAntdProps<AntDParagraphProps> {
    withoutMargins?: boolean;
    color?: TypographyColor;
}
export declare function Paragraph(userProps: TypographyParagraphProps): JSX.Element;
export {};
//# sourceMappingURL=Paragraph.d.ts.map