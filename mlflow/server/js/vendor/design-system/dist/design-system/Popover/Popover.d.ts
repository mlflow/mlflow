/// <reference types="react" />
import type { PopoverProps as AntDPopoverProps } from 'antd';
import type { DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
export interface PopoverProps extends Omit<AntDPopoverProps, 'content'>, DangerouslySetAntdProps<AntDPopoverProps>, HTMLDataAttributes {
    content?: React.ReactNode;
}
export declare const Popover: React.FC<PopoverProps>;
//# sourceMappingURL=Popover.d.ts.map