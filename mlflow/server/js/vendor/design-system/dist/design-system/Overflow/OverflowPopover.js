import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { useState } from 'react';
import { Button } from '../Button';
import { useDesignSystemTheme } from '../Hooks';
import { LegacyTooltip } from '../LegacyTooltip';
import { Popover } from '../Popover';
import { addDebugOutlineIfEnabled } from '../utils/debug';
export const OverflowPopover = ({ items, renderLabel, tooltipText, ariaLabel = 'More items', ...props }) => {
    const { theme } = useDesignSystemTheme();
    const [showTooltip, setShowTooltip] = useState(true);
    const label = `+${items.length}`;
    let trigger = (_jsx("span", { css: { lineHeight: 0 }, ...addDebugOutlineIfEnabled(), children: _jsx(Popover.Trigger, { asChild: true, children: _jsx(Button, { componentId: "something", type: "link", children: renderLabel ? renderLabel(label) : label }) }) }));
    if (showTooltip) {
        trigger = _jsx(LegacyTooltip, { title: tooltipText || 'See more items', children: trigger });
    }
    return (_jsxs(Popover.Root, { componentId: "codegen_design-system_src_design-system_overflow_overflowpopover.tsx_37", onOpenChange: (open) => setShowTooltip(!open), children: [trigger, _jsx(Popover.Content, { align: "start", "aria-label": ariaLabel, ...props, ...addDebugOutlineIfEnabled(), children: _jsx("div", { css: { display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }, children: items.map((item, index) => (_jsx("div", { children: item }, `overflow-${index}`))) }) })] }));
};
//# sourceMappingURL=OverflowPopover.js.map