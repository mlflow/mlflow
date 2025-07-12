import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import classnames from 'classnames';
import { useDesignSystemTheme } from '../Hooks';
import { InfoPopover } from '../Popover';
import { addDebugOutlineIfEnabled } from '../utils/debug';
const getLabelStyles = (theme, { inline }) => {
    const styles = {
        '&&': {
            color: theme.colors.textPrimary,
            fontWeight: theme.typography.typographyBoldFontWeight,
            display: inline ? 'inline' : 'block',
            lineHeight: theme.typography.lineHeightBase,
        },
    };
    return css(styles);
};
const getLabelWrapperStyles = (classNamePrefix, theme) => {
    const styles = {
        display: 'flex',
        gap: theme.spacing.xs,
        alignItems: 'center',
        [`&& + .${classNamePrefix}-input, && + .${classNamePrefix}-input-affix-wrapper, && + .${classNamePrefix}-select, && + .${classNamePrefix}-selectv2, && + .${classNamePrefix}-dialogcombobox, && + .${classNamePrefix}-checkbox-group, && + .${classNamePrefix}-radio-group, && + .${classNamePrefix}-typeahead-combobox, && + .${classNamePrefix}-datepicker, && + .${classNamePrefix}-rangepicker`]: {
            marginTop: theme.spacing.sm,
        },
    };
    return css(styles);
};
export const Label = (props) => {
    const { children, className, inline, required, infoPopoverContents, infoPopoverProps = {}, ...restProps } = props; // Destructure the new prop
    const { classNamePrefix, theme } = useDesignSystemTheme();
    const label = (_jsx("label", { ...addDebugOutlineIfEnabled(), css: [
            getLabelStyles(theme, { inline }),
            ...(!infoPopoverContents ? [getLabelWrapperStyles(classNamePrefix, theme)] : []),
        ], className: classnames(`${classNamePrefix}-label`, className), ...restProps, children: _jsxs("span", { css: { display: 'flex', alignItems: 'center' }, children: [children, required && _jsx("span", { "aria-hidden": "true", children: "*" })] }) }));
    return infoPopoverContents ? (_jsxs("div", { css: getLabelWrapperStyles(classNamePrefix, theme), children: [label, _jsx(InfoPopover, { ...infoPopoverProps, children: infoPopoverContents })] })) : (label);
};
//# sourceMappingURL=Label.js.map