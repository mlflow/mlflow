import { safex } from './safex';
export const useDesignSystemSafexFlags = () => {
    return {
        useNewShadows: safex('databricks.fe.designsystem.useNewShadows', false),
        useNewFormUISpacing: safex('databricks.fe.designsystem.useNewFormUISpacing', false),
        useNewBorderRadii: safex('databricks.fe.designsystem.useNewBorderRadii', false),
        useNewLargeAlertSizing: safex('databricks.fe.designsystem.useNewLargeAlertSizing', false),
        useNewBorderColors: safex('databricks.fe.designsystem.useNewBorderColors', false),
    };
};
//# sourceMappingURL=safex-flags.js.map