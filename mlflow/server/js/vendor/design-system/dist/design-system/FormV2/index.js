import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { FormMessage } from '../FormMessage/FormMessage';
import { Hint } from '../Hint/Hint';
import { useDesignSystemTheme } from '../Hooks';
import { Label } from '../Label/Label';
export * from './RHFAdapters';
const getHorizontalInputStyles = (theme, labelColWidth, inputColWidth) => {
    return css({
        display: 'flex',
        gap: theme.spacing.sm,
        '& > input, & > textarea, & > select': {
            marginTop: '0 !important',
        },
        '& > div:nth-of-type(1)': {
            width: labelColWidth,
        },
        '& > div:nth-of-type(2)': {
            width: inputColWidth,
        },
    });
};
const HorizontalFormRow = ({ children, labelColWidth = '33%', inputColWidth = '66%', ...restProps }) => {
    const { theme } = useDesignSystemTheme();
    return (_jsx("div", { css: getHorizontalInputStyles(theme, labelColWidth, inputColWidth), ...restProps, children: children }));
};
export const FormUI = {
    Message: FormMessage,
    Label: Label,
    Hint: Hint,
    HorizontalFormRow,
};
//# sourceMappingURL=index.js.map