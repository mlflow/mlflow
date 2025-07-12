import { Fragment as _Fragment, jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { Children } from 'react';
import { OverflowPopover } from './OverflowPopover';
import { useDesignSystemTheme } from '../Hooks';
import { Tag } from '../Tag';
export const Overflow = ({ children, noMargin = false, ...props }) => {
    const { theme } = useDesignSystemTheme();
    const childrenList = children && Children.toArray(children);
    if (!childrenList || childrenList.length === 0) {
        return _jsx(_Fragment, { children: children });
    }
    const firstItem = childrenList[0];
    const additionalItems = childrenList.splice(1);
    const renderOverflowLabel = (label) => (_jsx(Tag, { componentId: "codegen_design-system_src_design-system_overflow_overflow.tsx_28", css: getTagStyles(theme), children: label }));
    return additionalItems.length === 0 ? (_jsx(_Fragment, { children: firstItem })) : (_jsxs("div", { ...props, css: {
            display: 'inline-flex',
            alignItems: 'center',
            gap: noMargin ? 0 : theme.spacing.sm,
            maxWidth: '100%',
        }, children: [firstItem, additionalItems.length > 0 && (_jsx(OverflowPopover, { items: additionalItems, renderLabel: renderOverflowLabel, ...props }))] }));
};
const getTagStyles = (theme) => {
    const styles = {
        marginRight: 0,
        color: theme.colors.actionTertiaryTextDefault,
        cursor: 'pointer',
        '&:focus': {
            color: theme.colors.actionTertiaryTextDefault,
        },
        '&:hover': {
            color: theme.colors.actionTertiaryTextHover,
        },
        '&:active': {
            color: theme.colors.actionTertiaryTextPress,
        },
    };
    return css(styles);
};
//# sourceMappingURL=Overflow.js.map