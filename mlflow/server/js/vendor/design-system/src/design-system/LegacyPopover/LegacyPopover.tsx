import type { PopoverProps as AntDPopoverProps } from 'antd';
import { Popover as AntDPopover } from 'antd';

import { DesignSystemAntDConfigProvider, RestoreAntDDefaultClsPrefix } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks';
import type { DangerouslySetAntdProps, HTMLDataAttributes } from '../types';

/**
 * `LegacyPopover` is deprecated in favor of the new `Popover` component.
 * @deprecated
 */
export interface LegacyPopoverProps
  extends Omit<AntDPopoverProps, 'content'>,
    DangerouslySetAntdProps<AntDPopoverProps>,
    HTMLDataAttributes {
  content?: React.ReactNode;
}

/**
 * `LegacyPopover` is deprecated in favor of the new `Popover` component.
 * @deprecated
 */
export const LegacyPopover: React.FC<LegacyPopoverProps> = ({ content, dangerouslySetAntdProps, ...props }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <DesignSystemAntDConfigProvider>
      <AntDPopover
        zIndex={theme.options.zIndexBase + 30}
        {...props}
        content={<RestoreAntDDefaultClsPrefix>{content}</RestoreAntDDefaultClsPrefix>}
      />
    </DesignSystemAntDConfigProvider>
  );
};
