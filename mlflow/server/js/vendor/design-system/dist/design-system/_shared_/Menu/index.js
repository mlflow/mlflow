import { jsx as _jsx, Fragment as _Fragment, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { Children } from 'react';
import { HintColumn } from '../../DropdownMenu/DropdownMenu';
import { InfoIcon } from '../../Icon';
import { LegacyTooltip } from '../../LegacyTooltip';
const infoIconStyles = (theme) => ({
    display: 'inline-flex',
    paddingLeft: theme.spacing.xs,
    color: theme.colors.textSecondary,
    pointerEvents: 'all',
});
export const getNewChildren = (children, props, disabledReason, ref) => {
    const childCount = Children.count(children);
    const tooltip = (_jsx(LegacyTooltip, { title: disabledReason, placement: "right", dangerouslySetAntdProps: { getPopupContainer: () => ref.current || document.body }, children: _jsx("span", { "data-disabled-tooltip": true, css: (theme) => infoIconStyles(theme), onClick: (e) => {
                if (props.disabled) {
                    e.stopPropagation();
                }
            }, children: _jsx(InfoIcon, { role: "presentation", alt: "Disabled state reason", "aria-hidden": "false" }) }) }));
    if (childCount === 1) {
        return getChild(children, Boolean(props['disabled']), disabledReason, tooltip, 0, childCount);
    }
    return Children.map(children, (child, idx) => {
        return getChild(child, Boolean(props['disabled']), disabledReason, tooltip, idx, childCount);
    });
};
const getChild = (child, isDisabled, disabledReason, tooltip, index, siblingCount) => {
    const HintColumnType = (_jsx(HintColumn, {})).type;
    const isHintColumnType = Boolean(child &&
        typeof child !== 'string' &&
        typeof child !== 'number' &&
        typeof child !== 'boolean' &&
        'type' in child &&
        child?.type === HintColumnType);
    if (isDisabled && disabledReason && child && isHintColumnType) {
        return (_jsxs(_Fragment, { children: [tooltip, child] }));
    }
    else if (index === siblingCount - 1 && isDisabled && disabledReason) {
        return (_jsxs(_Fragment, { children: [child, tooltip] }));
    }
    return child;
};
//# sourceMappingURL=index.js.map