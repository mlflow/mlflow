import React, {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type PropsWithChildren,
  type SyntheticEvent,
} from 'react';

import { matchPredefinedError } from '../errors';
import type { HandleableError, PredefinedError } from '../errors';

export type UserActionError = PredefinedError | null;

type UserActionErrorHandlerContextProps = {
  currentUserActionError: UserActionError;
  handleError: (error: HandleableError, onErrorCallback?: (err: UserActionError) => void) => void;
  handlePromise: (promise: Promise<any>) => void;
  clearUserActionError: () => void;
};

const UserActionErrorHandlerContext = createContext<UserActionErrorHandlerContextProps>({
  currentUserActionError: null,
  handleError: () => {},
  handlePromise: () => {},
  clearUserActionError: () => {},
});

type UserActionErrorHandlerProps = {
  errorFilter?: (error: any) => boolean;
};

export const UserActionErrorHandler = ({ children, errorFilter }: PropsWithChildren<UserActionErrorHandlerProps>) => {
  const [currentUserActionError, setCurrentUserActionError] = useState<UserActionError>(null);

  const handleError = useCallback(
    (error: HandleableError, onErrorCallback?: (err: UserActionError) => void) => {
      if (!errorFilter?.(error)) {
        const predefinedError = matchPredefinedError(error);

        setCurrentUserActionError(predefinedError);

        if (onErrorCallback) {
          onErrorCallback(predefinedError);
        }
      }
    },
    [setCurrentUserActionError, errorFilter],
  );

  const handlePromise = useCallback(
    (promise: Promise<any>) => {
      promise.catch((error: HandleableError) => {
        handleError(error);
      });
    },
    [handleError],
  );

  const clearUserActionError = useCallback(() => {
    setCurrentUserActionError(null);
  }, [setCurrentUserActionError]);

  return (
    <UserActionErrorHandlerContext.Provider
      value={useMemo(
        () => ({
          currentUserActionError,
          handleError,
          handlePromise,
          clearUserActionError,
        }),
        [clearUserActionError, currentUserActionError, handleError, handlePromise],
      )}
    >
      {children}
    </UserActionErrorHandlerContext.Provider>
  );
};

export type UserErrorActionHandlerHook = {
  currentUserActionError: UserActionError;
  handleError: (error: HandleableError, onErrorCallback?: (err: UserActionError) => void) => void;
  /** @deprecated Use handleError instead, or get permission from #product-analytics to use */
  handleErrorWithEvent: (
    event: SyntheticEvent | Event,
    error: HandleableError,
    onErrorCallback?: (err: UserActionError) => void,
  ) => void;
  handlePromise: (promise: Promise<any>) => void;
  clearUserActionError: () => void;
};

export const useUserActionErrorHandler = (): UserErrorActionHandlerHook => {
  const { currentUserActionError, handleError, handlePromise, clearUserActionError } =
    useContext(UserActionErrorHandlerContext);

  const handleErrorWithEventImpl = useCallback(
    (event: SyntheticEvent | Event, error: HandleableError, onErrorCallback?: (err: UserActionError) => void) => {
      handleError(error, onErrorCallback);
    },
    [handleError],
  );

  return useMemo(
    () => ({
      currentUserActionError,
      handleError,
      handleErrorWithEvent: handleErrorWithEventImpl,
      handlePromise,
      clearUserActionError,
    }),
    [clearUserActionError, handleError, handlePromise, currentUserActionError, handleErrorWithEventImpl],
  );
};

export function withUserActionErrorHandler<P>(
  Component: React.ComponentType<React.PropsWithChildren<P>>,
  errorFilter?: (error: any) => boolean,
): React.ComponentType<React.PropsWithChildren<P>> {
  return function UserActionErrorHandlerWrapper(props: P) {
    return (
      <UserActionErrorHandler errorFilter={errorFilter}>
        {/* @ts-expect-error Generics don't play well with WithConditionalCSSProp type coming @emotion/react jsx typing to validate css= prop values typing. More details here: emotion-js/emotion#2169 */}
        <Component {...props} />
      </UserActionErrorHandler>
    );
  };
}
