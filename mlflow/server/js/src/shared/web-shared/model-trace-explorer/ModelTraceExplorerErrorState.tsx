import { DangerIcon, Empty } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

export const ModelTraceExplorerErrorState = () => {
  return (
    <Empty
      title={
        <FormattedMessage
          defaultMessage="Trace failed to render"
          description="Title for the error state in the model trace explorer"
        />
      }
      description={
        <FormattedMessage
          defaultMessage="Unfortunately, the trace could not be rendered due to an unknown error. You can reload the page to try again. If the problem persists, please contact support."
          description="Description for the error state in the model trace explorer"
        />
      }
      image={<DangerIcon />}
    />
  );
};
