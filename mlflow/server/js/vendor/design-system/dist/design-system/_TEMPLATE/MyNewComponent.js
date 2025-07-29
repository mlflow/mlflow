import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { Button as AntDButton } from 'antd';
import { DesignSystemAntDConfigProvider } from '../DesignSystemProvider';
export const MyNewComponent = (props) => {
    return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(AntDButton, { ...props }) }));
};
const defaultProps = {
    type: 'primary',
};
MyNewComponent.defaultProps = defaultProps;
//# sourceMappingURL=MyNewComponent.js.map