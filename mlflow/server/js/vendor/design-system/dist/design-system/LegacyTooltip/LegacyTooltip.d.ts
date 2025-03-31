import type { TooltipProps as AntDTooltipProps } from 'antd';
import React from 'react';
import type { HTMLDataAttributes } from '../types';
/**
 * `LegacyTooltip` is deprecated in favor of the new `Tooltip` component
 * @deprecated
 */
export interface LegacyTooltipProps extends HTMLDataAttributes {
    /**
     * The element with which the tooltip should be associated.
     */
    children: React.ReactNode;
    /**
     * Plain text that will appear within the tooltip. Links and formatted content should not be use. However, we allow
     * any React element to be passed in here, rather than just a string, to allow for i18n formatting components.
     */
    title: React.ReactNode;
    /**
     * Value that determines where the tooltip will be positioned relative to the associated element.
     */
    placement?: AntDTooltipProps['placement'];
    /**
     * Escape hatch to allow passing props directly to the underlying Ant `Tooltip` component.
     */
    dangerouslySetAntdProps?: Partial<AntDTooltipProps>;
    /**
     * ID used to refer to this element in unit tests.
     */
    dataTestId?: string;
    /**
     * Prop that forces the tooltip's arrow to be centered on the target element
     */
    arrowPointAtCenter?: boolean;
    /**
     * Toggle wrapper live region off
     */
    silenceScreenReader?: boolean;
    /**
     * Toggles screen readers reading the tooltip content as the label for the hovered/focused element
     */
    useAsLabel?: boolean;
}
/**
 * `LegacyTooltip` is deprecated in favor of the new `Tooltip` component
 * @deprecated
 */
export declare const LegacyTooltip: React.FC<LegacyTooltipProps>;
//# sourceMappingURL=LegacyTooltip.d.ts.map