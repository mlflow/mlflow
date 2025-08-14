import { ErrorBoundary } from 'react-error-boundary';
import { FormattedMessage } from '@databricks/i18n';
import { PageWrapper, Empty, DangerIcon } from '@databricks/design-system';
const PageFallback = ({ error }: { error?: Error }) => {
  return (
    <PageWrapper css={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <Empty
        data-testid="fallback"
        title={
          <FormattedMessage defaultMessage="Error" description="Title for error fallback component in Trace V3 page" />
        }
        description={
          error?.message ?? (
            <FormattedMessage
              defaultMessage="An error occurred while rendering this component."
              description="Description for default error message in Trace V3 page"
            />
          )
        }
        image={<DangerIcon />}
      />
    </PageWrapper>
  );
};

export const TracesV3PageWrapper = ({ children }: { children: React.ReactNode }) => {
  return <ErrorBoundary fallback={<PageFallback />}>{children}</ErrorBoundary>;
};
