import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import classnames from 'classnames';
import { useDesignSystemTheme } from '../Hooks';
import { addDebugOutlineIfEnabled } from '../utils/debug';
const getHintStyles = (classNamePrefix, theme) => {
    const styles = {
        display: 'block',
        color: theme.colors.textSecondary,
        lineHeight: theme.typography.lineHeightSm,
        fontSize: theme.typography.fontSizeSm,
        [`&& + .${classNamePrefix}-input, && + .${classNamePrefix}-input-affix-wrapper, && + .${classNamePrefix}-select, && + .${classNamePrefix}-selectv2, && + .${classNamePrefix}-dialogcombobox, && + .${classNamePrefix}-checkbox-group, && + .${classNamePrefix}-radio-group, && + .${classNamePrefix}-typeahead-combobox, && + .${classNamePrefix}-datepicker, && + .${classNamePrefix}-rangepicker`]: {
            marginTop: theme.spacing.sm,
        },
    };
    return css(styles);
};
export const Hint = (props) => {
    const { classNamePrefix, theme } = useDesignSystemTheme();
    const { className, ...restProps } = props;
    return (_jsx("span", { ...addDebugOutlineIfEnabled(), className: classnames(`${classNamePrefix}-hint`, className), css: getHintStyles(classNamePrefix, theme), ...restProps }));
};
//# sourceMappingURL=Hint.js.map