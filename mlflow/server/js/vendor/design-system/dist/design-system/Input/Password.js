import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { Input as AntDInput } from 'antd';
import { forwardRef } from 'react';
import { getInputEmotionStyles } from './Input';
import { DesignSystemAntDConfigProvider } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks/useDesignSystemTheme';
import { useDesignSystemSafexFlags } from '../utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
export const Password = forwardRef(function Password({ validationState, autoComplete = 'off', dangerouslySetAntdProps, dangerouslyAppendEmotionCSS, ...props }, ref) {
    const { classNamePrefix, theme } = useDesignSystemTheme();
    const { useNewShadows, useNewBorderRadii } = useDesignSystemSafexFlags();
    return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(AntDInput.Password, { ...addDebugOutlineIfEnabled(), visibilityToggle: false, ref: ref, autoComplete: autoComplete, css: [
                getInputEmotionStyles(classNamePrefix, theme, { validationState, useNewShadows, useNewBorderRadii }),
                dangerouslyAppendEmotionCSS,
            ], ...props, ...dangerouslySetAntdProps }) }));
});
//# sourceMappingURL=Password.js.map