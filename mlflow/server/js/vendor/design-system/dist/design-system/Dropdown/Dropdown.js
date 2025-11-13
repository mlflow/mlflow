import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { Dropdown as AntDDropdown } from 'antd';
import { DesignSystemAntDConfigProvider } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks';
import { addDebugOutlineIfEnabled } from '../utils/debug';
/**
 * @deprecated Use `DropdownMenu` instead.
 */
export const Dropdown = ({ dangerouslySetAntdProps, ...props }) => {
    const { theme } = useDesignSystemTheme();
    return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(AntDDropdown, { ...addDebugOutlineIfEnabled(), mouseLeaveDelay: 0.25, ...props, overlayStyle: {
                zIndex: theme.options.zIndexBase + 50,
                ...props.overlayStyle,
            }, ...dangerouslySetAntdProps }) }));
};
//# sourceMappingURL=Dropdown.js.map