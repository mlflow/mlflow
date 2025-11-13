import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { Row as AntDRow } from 'antd';
import { DesignSystemAntDConfigProvider } from '../DesignSystemProvider';
export const ROW_GUTTER_SIZE = 8;
export const Row = ({ gutter = ROW_GUTTER_SIZE, ...props }) => {
    return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(AntDRow, { gutter: gutter, ...props }) }));
};
//# sourceMappingURL=Row.js.map