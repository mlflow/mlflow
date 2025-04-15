import { UserActionErrorHandler } from '@databricks/web-shared/metrics';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { ErrorBoundary } from 'react-error-boundary';
import { DangerIcon, Empty, PageWrapper } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

const PageFallback = ({ error }: { error?: Error }) => {
  return (
    <PageWrapper css={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <Empty
        data-testid="fallback"
        title={
          <FormattedMessage
            defaultMessage="Error"
            description="Title for error fallback component in prompts management UI"
          />
        }
        description={
          error?.message ?? (
            <FormattedMessage
              defaultMessage="An error occurred while rendering this component."
              description="Description for default error message in prompts management UI"
            />
          )
        }
        image={<DangerIcon />}
      />
    </PageWrapper>
  );
};

/**
 * Wrapper for all experiment logged model pages.
 * Provides error boundaries and user action error handling.
 */
export const ExperimentLoggedModelPageWrapper = ({
  children,
  resetKey,
}: {
  children: React.ReactNode;
  resetKey?: unknown;
}) => {
  return (
    <ErrorBoundary FallbackComponent={PageFallback} resetKeys={[resetKey]}>
      <UserActionErrorHandler>{children}</UserActionErrorHandler>
    </ErrorBoundary>
  );
};
