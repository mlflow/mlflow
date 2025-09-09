import React from 'react';
import type { ErrorBoundaryPropsWithComponent, FallbackProps } from 'react-error-boundary';
import { ErrorBoundary } from 'react-error-boundary';
import ErrorUtils from './ErrorUtils';
import { DangerIcon, Empty } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

export type ErrorBoundaryProps = {
  children: React.Component;
  customFallbackComponent?: ErrorBoundaryPropsWithComponent['FallbackComponent'];
};

function ErrorFallback() {
  return (
    <Empty
      data-testid="fallback"
      title={<FormattedMessage defaultMessage="Error" description="Title of editor error fallback component" />}
      description={
        <FormattedMessage
          defaultMessage="An error occurred while rendering this component."
          description="Description of error fallback component"
        />
      }
      image={<DangerIcon />}
    />
  );
}

function CustomErrorBoundary({ children, customFallbackComponent }: React.PropsWithChildren<ErrorBoundaryProps>) {
  function logErrorToConsole(error: Error, info: { componentStack: string }) {
    // eslint-disable-next-line no-console -- TODO(FEINF-3587)
    console.error('Caught Unexpected Error: ', error, info.componentStack);
  }

  if (customFallbackComponent) {
    return (
      <ErrorBoundary onError={logErrorToConsole} FallbackComponent={customFallbackComponent}>
        {children}
      </ErrorBoundary>
    );
  }

  return (
    <ErrorBoundary onError={logErrorToConsole} fallback={<ErrorFallback />}>
      {children}
    </ErrorBoundary>
  );
}

export function withErrorBoundary<P>(
  service: string,
  Component: React.ComponentType<React.PropsWithChildren<P>>,
  errorMessage?: React.ReactNode,
  customFallbackComponent?: React.ComponentType<React.PropsWithChildren<FallbackProps>>,
): React.ComponentType<React.PropsWithChildren<P>> {
  return function CustomErrorBoundaryWrapper(props: P) {
    return (
      <CustomErrorBoundary customFallbackComponent={customFallbackComponent}>
        {/* @ts-expect-error Generics don't play well with WithConditionalCSSProp type coming @emotion/react jsx typing to validate css= prop values typing. More details here: emotion-js/emotion#2169 */}
        <Component {...props} />
      </CustomErrorBoundary>
    );
  };
}
