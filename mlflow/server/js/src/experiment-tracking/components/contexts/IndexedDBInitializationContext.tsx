import React, { createContext, useEffect, useState } from 'react';
import { globalIndexedDBStorage } from '@mlflow/mlflow/src/common/utils/LocalStorageUtils';
import { Spinner } from '@databricks/design-system';
import Utils from '@mlflow/mlflow/src/common/utils/Utils';

export const IndexedDBInitializationContext = createContext<{
  isIndexedDBAvailable?: boolean;
}>({});

export function IndexedDBInitializationContextProvider(props: React.PropsWithChildren<{}>) {
  const [ready, setReady] = useState<boolean>(false);
  const [isIndexedDBAvailable, setIsIndexedDBAvailable] = useState<boolean>(false);

  useEffect(() => {
    (async () => {
      try {
        await globalIndexedDBStorage.initialize();
        setIsIndexedDBAvailable(true);
      } catch (error) {
        Utils.logErrorAndNotifyUser(
          `IndexedDB unavailable - using browser local storage instead. Your settings will still be saved. Error: ${error}`,
        );
        setIsIndexedDBAvailable(false);
      } finally {
        setReady(true);
      }
    })();
  }, []);

  if (!ready) {
    return <Spinner />;
  }

  return (
    <IndexedDBInitializationContext.Provider value={{ isIndexedDBAvailable }}>
      {props.children}
    </IndexedDBInitializationContext.Provider>
  );
}
