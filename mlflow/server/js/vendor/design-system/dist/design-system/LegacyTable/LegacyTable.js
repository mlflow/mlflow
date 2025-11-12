import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { Table as AntDTable } from 'antd';
import { DesignSystemAntDConfigProvider, RestoreAntDDefaultClsPrefix } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks/useDesignSystemTheme';
import { getPaginationEmotionStyles } from '../Pagination';
import { Spinner } from '../Spinner';
import { useDesignSystemSafexFlags } from '../utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
const getTableEmotionStyles = (classNamePrefix, theme, scrollableInFlexibleContainer, useNewShadows, useNewBorderColors) => {
    const styles = [
        css({
            [`.${classNamePrefix}-table-pagination`]: {
                ...getPaginationEmotionStyles(classNamePrefix, theme, useNewShadows, useNewBorderColors),
            },
        }),
    ];
    if (scrollableInFlexibleContainer) {
        styles.push(getScrollableInFlexibleContainerStyles(classNamePrefix));
    }
    return styles;
};
const getScrollableInFlexibleContainerStyles = (clsPrefix) => {
    const styles = {
        minHeight: 0,
        [`.${clsPrefix}-spin-nested-loading`]: { height: '100%' },
        [`.${clsPrefix}-spin-container`]: { height: '100%', display: 'flex', flexDirection: 'column' },
        [`.${clsPrefix}-table-container`]: { height: '100%', display: 'flex', flexDirection: 'column' },
        [`.${clsPrefix}-table`]: { minHeight: 0 },
        [`.${clsPrefix}-table-header`]: { flexShrink: 0 },
        [`.${clsPrefix}-table-body`]: { minHeight: 0 },
    };
    return css(styles);
};
const DEFAULT_LOADING_SPIN_PROPS = { indicator: _jsx(Spinner, {}) };
/**
 * `LegacyTable` is deprecated in favor of the new `Table` component
 * For more information please join #dubois-table-early-adopters in Slack.
 * @deprecated
 */
// eslint-disable-next-line @typescript-eslint/ban-types
export const LegacyTable = (props) => {
    const { classNamePrefix, theme } = useDesignSystemTheme();
    const { useNewShadows, useNewBorderColors } = useDesignSystemSafexFlags();
    const { loading, scrollableInFlexibleContainer, children, ...tableProps } = props;
    return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(AntDTable, { ...addDebugOutlineIfEnabled(), 
            // NOTE(FEINF-1273): The default loading indicator from AntD does not animate
            // and the design system spinner is recommended over the AntD one. Therefore,
            // if `loading` is `true`, render the design system <Spinner/> component.
            loading: loading === true ? DEFAULT_LOADING_SPIN_PROPS : loading, scroll: scrollableInFlexibleContainer ? { y: 'auto' } : undefined, ...tableProps, css: getTableEmotionStyles(classNamePrefix, theme, Boolean(scrollableInFlexibleContainer), useNewShadows, useNewBorderColors), 
            // ES-902549 this allows column names of "children", using a name that is less likely to be hit
            expandable: { childrenColumnName: '__antdChildren' }, children: _jsx(RestoreAntDDefaultClsPrefix, { children: children }) }) }));
};
//# sourceMappingURL=LegacyTable.js.map