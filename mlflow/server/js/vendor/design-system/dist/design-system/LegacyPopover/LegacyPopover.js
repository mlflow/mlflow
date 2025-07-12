import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { Popover as AntDPopover } from 'antd';
import { DesignSystemAntDConfigProvider, RestoreAntDDefaultClsPrefix } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks';
/**
 * `LegacyPopover` is deprecated in favor of the new `Popover` component.
 * @deprecated
 */
export const LegacyPopover = ({ content, dangerouslySetAntdProps, ...props }) => {
    const { theme } = useDesignSystemTheme();
    return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(AntDPopover, { zIndex: theme.options.zIndexBase + 30, ...props, content: _jsx(RestoreAntDDefaultClsPrefix, { children: content }) }) }));
};
//# sourceMappingURL=LegacyPopover.js.map