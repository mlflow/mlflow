import type { TypographyTextProps } from './Text';
export type TypographyTextMiddleElideProps = Omit<TypographyTextProps, 'ellipsis' | 'children'> & {
    text: string;
    suffixLength?: number;
};
export declare function TextMiddleElide({ text, suffixLength, ...textProps }: TypographyTextMiddleElideProps): JSX.Element;
export declare function getStartAndSuffix(text: string, suffixLength: number): {
    start: string;
    suffix: undefined;
} | {
    start: string;
    suffix: string;
};
//# sourceMappingURL=TextMiddleElide.d.ts.map