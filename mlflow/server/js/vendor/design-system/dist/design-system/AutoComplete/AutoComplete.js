import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { AutoComplete as AntDAutoComplete } from 'antd';
import { DesignSystemAntDConfigProvider, getAnimationCss } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks/useDesignSystemTheme';
import { getDarkModePortalStyles, useDesignSystemSafexFlags } from '../utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
/**
 * @deprecated Use `TypeaheadCombobox` instead.
 */
export const AutoComplete = /* #__PURE__ */ (() => {
    const AutoComplete = ({ dangerouslySetAntdProps, ...props }) => {
        const { theme } = useDesignSystemTheme();
        const { useNewShadows, useNewBorderColors } = useDesignSystemSafexFlags();
        return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(AntDAutoComplete, { ...addDebugOutlineIfEnabled(), dropdownStyle: {
                    boxShadow: theme.general.shadowLow,
                    ...getDarkModePortalStyles(theme, useNewShadows, useNewBorderColors),
                }, ...props, ...dangerouslySetAntdProps, css: css(getAnimationCss(theme.options.enableAnimation)) }) }));
    };
    AutoComplete.Option = AntDAutoComplete.Option;
    return AutoComplete;
})();
//# sourceMappingURL=AutoComplete.js.map