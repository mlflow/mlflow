import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { Typography as AntDTypography } from 'antd';
import { Hint } from './Hint';
import { Link } from './Link';
import { Paragraph } from './Paragraph';
import { Text } from './Text';
import { TextMiddleElide } from './TextMiddleElide';
import { Title } from './Title';
import { Truncate } from './Truncate';
import { DesignSystemAntDConfigProvider } from '../DesignSystemProvider';
import { addDebugOutlineIfEnabled } from '../utils/debug';
export const Typography = /* #__PURE__ */ (() => {
    function Typography({ dangerouslySetAntdProps, ...props }) {
        return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(AntDTypography, { ...addDebugOutlineIfEnabled(), ...props, ...dangerouslySetAntdProps }) }));
    }
    Typography.Text = Text;
    Typography.Title = Title;
    Typography.Paragraph = Paragraph;
    Typography.Link = Link;
    Typography.Hint = Hint;
    Typography.Truncate = Truncate;
    Typography.TextMiddleElide = TextMiddleElide;
    return Typography;
})();
//# sourceMappingURL=Typography.js.map