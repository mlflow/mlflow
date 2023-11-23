import React, { useContext, useEffect, useMemo, useState } from 'react';
import { shouldUseToggleModelsNextUI } from '../../common/utils/FeatureUtils';

const localStorageKey = '_mlflow_user_setting_use_next_models_ui';

const NextModelsUIContext = React.createContext<{
  usingNextModelsUI: boolean;
  setUsingNextModelsUI: (newValue: boolean) => void;
}>({
  usingNextModelsUI: false,
  setUsingNextModelsUI: () => {},
});

/**
 * Get
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
  <P,>(Component: React.ComponentType<P & { usingNextModelsUI?: boolean }>) =>
  (props: P) => {
    const [usingNextModelsUI, setUsingNextModelsUI] = useState(
      localStorage.getItem(localStorageKey) === 'true',
    );

    useEffect(() => {
      localStorage.setItem(localStorageKey, usingNextModelsUI.toString());
    }, [usingNextModelsUI]);

    const contextValue = useMemo(
      () => ({ usingNextModelsUI, setUsingNextModelsUI }),
      [usingNextModelsUI],
    );

    if (!shouldUseToggleModelsNextUI()) {
      return <Component {...props} usingNextModelsUI={false} />;
    }

    return (
      <NextModelsUIContext.Provider value={contextValue}>
        <Component {...props} usingNextModelsUI={contextValue.usingNextModelsUI} />
      </NextModelsUIContext.Provider>
    );
  };
