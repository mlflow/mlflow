import { safex } from './safex';

export const useDesignSystemSafexFlags = () => {
  return {
    useNewShadows: safex('databricks.fe.designsystem.useNewShadows', false),
    useNewFormUISpacing: safex('databricks.fe.designsystem.useNewFormUISpacing', false),
  };
};
