import { FormattedMessage } from 'react-intl';
import { EXPERIMENT_PAGE_FEEDBACK_URL } from '@mlflow/mlflow/src/experiment-tracking/constants';

export const getShareFeedbackOverflowMenuItem = () => {
  return {
    id: 'feedback',
    itemName: (
      <FormattedMessage
        defaultMessage="Send Feedback"
        description="Text for provide feedback button on experiment view page header"
      />
    ),
    href: EXPERIMENT_PAGE_FEEDBACK_URL,
  };
};
