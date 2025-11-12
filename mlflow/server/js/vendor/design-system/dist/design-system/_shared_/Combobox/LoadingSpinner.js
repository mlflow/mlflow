import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { useDesignSystemTheme } from '../../Hooks';
import { Spinner } from '../../Spinner';
export const LoadingSpinner = (props) => {
    const { theme } = useDesignSystemTheme();
    return (_jsx(Spinner, { css: {
            display: 'flex',
            alignSelf: 'center',
            justifyContent: 'center',
            alignItems: 'center',
            height: theme.general.heightSm,
            width: theme.general.heightSm,
            '> span': { fontSize: 20 },
        }, ...props }));
};
//# sourceMappingURL=LoadingSpinner.js.map