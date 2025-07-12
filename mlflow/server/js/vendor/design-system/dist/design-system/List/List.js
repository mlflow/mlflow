import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { List as AntDList } from 'antd';
import { DesignSystemAntDConfigProvider } from '../DesignSystemProvider';
import { addDebugOutlineIfEnabled } from '../utils/debug';
export const List = /* #__PURE__ */ (() => {
    function List({ ...props }) {
        return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(AntDList, { ...addDebugOutlineIfEnabled(), ...props }) }));
    }
    List.Item = AntDList.Item;
    return List;
})();
//# sourceMappingURL=List.js.map