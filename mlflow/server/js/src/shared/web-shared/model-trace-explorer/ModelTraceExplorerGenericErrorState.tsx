import { DangerIcon, Empty, PageWrapper } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

export const ModelTraceExplorerGenericErrorState = ({ error }: { error?: Error }) => {
  return (
    <PageWrapper css={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <Empty
        data-testid="fallback"
        title={
          <FormattedMessage
            defaultMessage="Error"
            description="Title for error fallback component in Trace Explorer UI"
          />
        }
        description={
          error?.message ?? (
            <FormattedMessage
              defaultMessage="An error occurred while rendering the trace"
              description="Description for default error message in the Trace Explorer UI"
            />
          )
        }
        image={<DangerIcon />}
      />
    </PageWrapper>
  );
};
