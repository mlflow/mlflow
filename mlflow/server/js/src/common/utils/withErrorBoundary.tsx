import React from 'react';
import { ErrorBoundary } from 'react-error-boundary';
import ErrorUtils from './ErrorUtils';
import { DangerIcon, Empty } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

export type ErrorBoundaryProps = {
  children: React.Component;
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

export function CustomErrorBoundary({ children }: React.PropsWithChildren<ErrorBoundaryProps>) {
  function logErrorToConsole(error: Error, info: { componentStack: string }) {
    console.error('Caught Unexpected Error: ', error, info.componentStack);
  }

  return (
    <ErrorBoundary onError={logErrorToConsole} fallback={<ErrorFallback />}>
      {children}
    </ErrorBoundary>
  );
}

export function withErrorBoundary<P>(
  service: string,
  Component: React.ComponentType<P>,
  errorMessage?: React.ReactNode,
): React.ComponentType<P> {
  return function CustomErrorBoundaryWrapper(props: P) {
    return (
      <CustomErrorBoundary>
        {/* @ts-expect-error Generics don't play well with WithConditionalCSSProp type coming @emotion/react jsx typing to validate css= prop values typing. More details here: emotion-js/emotion#2169 */}
        <Component {...props} />
      </CustomErrorBoundary>
    );
  };
}
