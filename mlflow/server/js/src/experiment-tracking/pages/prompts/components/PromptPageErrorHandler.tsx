import { DangerIcon, Empty, PageWrapper } from '@databricks/design-system';
import type { FallbackProps } from 'react-error-boundary';
import { FormattedMessage } from 'react-intl';
import { ScrollablePageWrapperStyles } from '../../../../common/components/ScrollablePageWrapper';

export const PromptPageErrorHandler = ({ error }: { error?: Error }) => {
  return (
    <PageWrapper
      css={{ flex: 1, ...ScrollablePageWrapperStyles, display: 'flex', alignItems: 'center', justifyContent: 'center' }}
    >
      <Empty
        data-testid="fallback"
        title={<FormattedMessage defaultMessage="Error" description="Title of editor error fallback component" />}
        description={
          error?.message ?? (
            <FormattedMessage
              defaultMessage="An error occurred while rendering this component."
              description="Description of error fallback component"
            />
          )
        }
        image={<DangerIcon />}
      />
    </PageWrapper>
  );
};
