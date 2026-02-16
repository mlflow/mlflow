import { DangerIcon, Empty, PageWrapper } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

export const TracesV3GenericErrorState = ({ error }: { error?: Error }) => {
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
