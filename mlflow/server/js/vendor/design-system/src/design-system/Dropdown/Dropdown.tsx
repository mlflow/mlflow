import type { DropDownProps as AntDDropdownProps } from 'antd';
import { Dropdown as AntDDropdown } from 'antd';

import { DesignSystemAntDConfigProvider } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks';
import type { DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
import { addDebugOutlineIfEnabled } from '../utils/debug';

export interface DropdownProps
  extends AntDDropdownProps,
    DangerouslySetAntdProps<AntDDropdownProps>,
    HTMLDataAttributes {}

/**
 * @deprecated Use `DropdownMenu` instead.
 */
export const Dropdown: React.FC<DropdownProps> = ({ dangerouslySetAntdProps, ...props }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <DesignSystemAntDConfigProvider>
      <AntDDropdown
        {...addDebugOutlineIfEnabled()}
        mouseLeaveDelay={0.25}
        {...props}
        overlayStyle={{
          zIndex: theme.options.zIndexBase + 50,
          ...props.overlayStyle,
        }}
        {...dangerouslySetAntdProps}
      />
    </DesignSystemAntDConfigProvider>
  );
};
