import { UserActionErrorHandler } from '@databricks/web-shared/metrics';
import { ErrorBoundary } from 'react-error-boundary';
import { DangerIcon, Empty, PageWrapper } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { QueryClientProvider, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';

const PageFallback = ({ error }: { error?: Error }) => {
  return (
    <PageWrapper css={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <Empty
        data-testid="fallback"
        title={
          <FormattedMessage
            defaultMessage="Error"
            description="Title for error fallback component in experiment evaluation runs UI"
          />
        }
        description={
          error?.message ?? (
            <FormattedMessage
              defaultMessage="An error occurred while rendering this component."
              description="Description for default error message in experiment evaluation runs UI"
            />
          )
        }
        image={<DangerIcon />}
      />
    </PageWrapper>
  );
};

/**
 * Wrapper for all experiment evaluation runs pages.
 * Provides error boundaries and user action error handling.
 */
export const ExperimentEvaluationRunsPageWrapper = ({
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
