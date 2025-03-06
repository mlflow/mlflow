import React, { useContext, useEffect, useMemo, useState } from 'react';
import { shouldShowModelsNextUI } from '../../common/utils/FeatureUtils';

const useOldModelsUIStorageKey = '_mlflow_user_setting_dismiss_next_model_registry_ui';

const NextModelsUIContext = React.createContext<{
  usingNextModelsUI: boolean;
  setUsingNextModelsUI: (newValue: boolean) => void;
}>({
  usingNextModelsUI: shouldShowModelsNextUI(),
  setUsingNextModelsUI: () => {},
});

/**
 * Get the current context value for the next models UI.
 */
export const useNextModelsUIContext = () => useContext(NextModelsUIContext);

/**
 * Wraps the component with tools allowing to get and change the current value of
 * "use next models UI" toggle flag. It will wrap the component with the relevant React Context
 * and in order to make it usable in class components, it also injects `usingNextModelsUI`
 * boolean prop with the current flag value. To easily access the context in the downstream
 * function components, use `useNextModelsUIContext()` hook.
 */
export const withNextModelsUIContext =
  <BaseProps extends { usingNextModelsUI?: boolean }>(
    Component: React.ComponentType<BaseProps>,
  ) =>
  (props: Omit<BaseProps, 'usingNextModelsUI'>) => {
    const [usingNextModelsUI, setUsingNextModelsUI] = useState(
      localStorage.getItem(useOldModelsUIStorageKey) !== 'true',
    );

    useEffect(() => {
      localStorage.setItem(useOldModelsUIStorageKey, (!usingNextModelsUI).toString());
    }, [usingNextModelsUI]);

    const contextValue = useMemo(() => ({ usingNextModelsUI, setUsingNextModelsUI }), [usingNextModelsUI]);

    if (!shouldShowModelsNextUI()) {
      return <Component {...(props as BaseProps)} usingNextModelsUI={false} />;
    }

    return (
      <NextModelsUIContext.Provider value={contextValue}>
        <Component {...(props as BaseProps)} usingNextModelsUI={contextValue.usingNextModelsUI} />
      </NextModelsUIContext.Provider>
    );
  };