import { DangerIcon, Empty, PageWrapper } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

export const ExperimentChatSessionsGenericErrorState = ({ error }: { error?: Error }) => {
  return (
    <PageWrapper css={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <Empty
        data-testid="fallback"
        title={
          <FormattedMessage
            defaultMessage="Error"
            description="Title for error fallback component in the MLflow experiment chat sessions page"
          />
        }
        description={
          error?.message ?? (
            <FormattedMessage
              defaultMessage="An error occurred while rendering chat sessions."
              description="Description for default error message in the MLflow experiment chat sessions page"
            />
          )
        }
        image={<DangerIcon />}
      />
    </PageWrapper>
  );
};
