import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { Col as AntDCol } from 'antd';
import { DesignSystemAntDConfigProvider } from '../DesignSystemProvider';
export const Col = ({ dangerouslySetAntdProps, ...props }) => (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(AntDCol, { ...props, ...dangerouslySetAntdProps }) }));
//# sourceMappingURL=Col.js.map