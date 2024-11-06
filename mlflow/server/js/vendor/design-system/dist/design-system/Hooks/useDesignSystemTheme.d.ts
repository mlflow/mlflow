import type { ReactElement } from 'react';
import React from 'react';
import type { Theme } from '../../theme';
export interface DesignSystemThemeInterface {
    /** Theme object that contains colors, spacing, and other variables **/
    theme: Theme;
    /** Prefix that is used in front of each className.  **/
    classNamePrefix: string;
    /** Helper method that use be used to construct the full className of an underlying AntD className.
     * Use with caution and prefer emotion.js when possible **/
    getPrefixedClassName: (className: string) => string;
}
export declare function getClassNamePrefix(theme: Theme): string;
export declare function getPrefixedClassNameFromTheme(theme: Theme, className: string | null | undefined): string;
export declare function useDesignSystemTheme(): DesignSystemThemeInterface;
export type DesignSystemHocProps = {
    designSystemThemeApi: DesignSystemThemeInterface;
};
export declare function WithDesignSystemThemeHoc<P>(Component: React.ComponentType<P & DesignSystemHocProps>): (props: P) => ReactElement;
//# sourceMappingURL=useDesignSystemTheme.d.ts.map