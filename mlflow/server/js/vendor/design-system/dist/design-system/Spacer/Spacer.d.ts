import type { HTMLDataAttributes } from '../types';
type validSpacerOptions = 'xs' | 'sm' | 'md' | 'lg';
export interface SpacerProps extends HTMLDataAttributes {
    size?: validSpacerOptions;
    /** Prevents the Spacer component from shrinking when used in flexbox columns. **/
    shrinks?: boolean;
}
export declare const Spacer: React.FC<SpacerProps>;
export {};
//# sourceMappingURL=Spacer.d.ts.map