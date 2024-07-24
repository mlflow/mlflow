import type { PopoverProps as AntDPopoverProps } from 'antd';
import type { DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
/**
 * `LegacyPopover` is deprecated in favor of the new `Popover` component.
 * @deprecated
 */
export interface LegacyPopoverProps extends Omit<AntDPopoverProps, 'content'>, DangerouslySetAntdProps<AntDPopoverProps>, HTMLDataAttributes {
    content?: React.ReactNode;
}
/**
 * `LegacyPopover` is deprecated in favor of the new `Popover` component.
 * @deprecated
 */
export declare const LegacyPopover: React.FC<LegacyPopoverProps>;
//# sourceMappingURL=LegacyPopover.d.ts.map