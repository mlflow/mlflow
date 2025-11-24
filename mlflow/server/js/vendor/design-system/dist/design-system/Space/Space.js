import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { Space as AntDSpace } from 'antd';
import { DesignSystemAntDConfigProvider } from '../DesignSystemProvider';
export const Space = ({ dangerouslySetAntdProps, ...props }) => {
    return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(AntDSpace, { ...props, ...dangerouslySetAntdProps }) }));
};
//# sourceMappingURL=Space.js.map