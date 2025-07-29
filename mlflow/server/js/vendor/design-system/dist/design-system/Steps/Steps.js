import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { Steps as AntDSteps } from 'antd';
import { DesignSystemAntDConfigProvider } from '../DesignSystemProvider';
import { addDebugOutlineIfEnabled } from '../utils/debug';
/** @deprecated Please use the supported Stepper widget instead. See https://ui-infra.dev.databricks.com/storybook/js/packages/du-bois/index.html?path=/docs/primitives-stepper--docs */
export const Steps = /* #__PURE__ */ (() => {
    function Steps({ dangerouslySetAntdProps, ...props }) {
        return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(AntDSteps, { ...addDebugOutlineIfEnabled(), ...props, ...dangerouslySetAntdProps }) }));
    }
    Steps.Step = AntDSteps.Step;
    return Steps;
})();
//# sourceMappingURL=Steps.js.map