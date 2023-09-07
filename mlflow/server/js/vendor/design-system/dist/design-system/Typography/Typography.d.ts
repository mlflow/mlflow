import type { TypographyProps as AntDTypographyProps } from 'antd';
import type { ReactNode } from 'react';
import type { DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
import { Link } from './Link';
import { Paragraph } from './Paragraph';
import { Text } from './Text';
import { Title } from './Title';
interface TypographyProps extends Omit<AntDTypographyProps, 'Text' | 'Title' | 'Paragraph' | 'Link'>, DangerouslySetAntdProps<AntDTypographyProps>, HTMLDataAttributes {
    children?: ReactNode;
    withoutMargins?: boolean;
}
export declare const Typography: {
    ({ dangerouslySetAntdProps, ...props }: TypographyProps): JSX.Element;
    Text: typeof Text;
    Title: typeof Title;
    Paragraph: typeof Paragraph;
    Link: typeof Link;
};
export {};
//# sourceMappingURL=Typography.d.ts.map