import { DangerIcon, Empty } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { ScrollablePageWrapper } from '../../../../common/components/ScrollablePageWrapper';

export const PromptPageErrorHandler = ({ error }: { error?: Error }) => {
  return (
    <ScrollablePageWrapper css={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
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
    </ScrollablePageWrapper>
  );
};
