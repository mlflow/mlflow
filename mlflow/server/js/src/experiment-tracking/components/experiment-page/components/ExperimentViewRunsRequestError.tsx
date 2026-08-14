import { Empty, WarningIcon } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

export const ExperimentViewRunsRequestError = ({ error }: { error?: Error | null }) => {
  const message = error?.message;

  return (
    <div css={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <Empty
        description={
          message ?? (
            <FormattedMessage
              defaultMessage="Your request could not be fulfilled. Please try again."
              description="A message shown on the experiment page if the runs request fails"
            />
          )
        }
        image={<WarningIcon />}
        title={
          <FormattedMessage
            defaultMessage="Request error"
            description="A title shown on the experiment page if the runs request fails"
          />
        }
      />
    </div>
  );
};
