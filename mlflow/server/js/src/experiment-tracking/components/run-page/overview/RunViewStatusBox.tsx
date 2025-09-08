import { Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';
import type { RunInfoEntity } from '../../../types';
import { RunStatusIcon } from '../../RunStatusIcon';
import { FormattedMessage } from 'react-intl';
import type { MlflowRunStatus } from '../../../../graphql/__generated__/graphql';

/**
 * Displays run status cell in run detail overview.
 */
export const RunViewStatusBox = ({ status }: { status: RunInfoEntity['status'] | MlflowRunStatus | null }) => {
  const { theme } = useDesignSystemTheme();
  const getTagColor = () => {
    if (status === 'FINISHED') {
      return theme.isDarkMode ? theme.colors.green800 : theme.colors.green100;
    }
    if (status === 'KILLED' || status === 'FAILED') {
      return theme.isDarkMode ? theme.colors.red800 : theme.colors.red100;
    }
    if (status === 'SCHEDULED' || status === 'RUNNING') {
      return theme.isDarkMode ? theme.colors.blue800 : theme.colors.blue100;
    }

    return undefined;
  };

  const getStatusLabel = () => {
    if (status === 'FINISHED') {
      return (
        <Typography.Text color="success">
          <FormattedMessage
            defaultMessage="Finished"
            description="Run page > Overview > Run status cell > Value for finished state"
          />
        </Typography.Text>
      );
    }
    if (status === 'KILLED') {
      return (
        <Typography.Text color="error">
          <FormattedMessage
            defaultMessage="Killed"
            description="Run page > Overview > Run status cell > Value for killed state"
          />
        </Typography.Text>
      );
    }
    if (status === 'FAILED') {
      return (
        <Typography.Text color="error">
          <FormattedMessage
            defaultMessage="Failed"
            description="Run page > Overview > Run status cell > Value for failed state"
          />
        </Typography.Text>
      );
    }
    if (status === 'RUNNING') {
      return (
        <Typography.Text color="info">
          <FormattedMessage
            defaultMessage="Running"
            description="Run page > Overview > Run status cell > Value for running state"
          />
        </Typography.Text>
      );
    }
    if (status === 'SCHEDULED') {
      return (
        <Typography.Text color="info">
          <FormattedMessage
            defaultMessage="Scheduled"
            description="Run page > Overview > Run status cell > Value for scheduled state"
          />
        </Typography.Text>
      );
    }
    return status;
  };

  return (
    <Tag
      componentId="codegen_mlflow_app_src_experiment-tracking_components_run-page_overview_runviewstatusbox.tsx_81"
      css={{ backgroundColor: getTagColor() }}
    >
      {status && <RunStatusIcon status={status} />}{' '}
      <Typography.Text css={{ marginLeft: theme.spacing.sm }}>{getStatusLabel()}</Typography.Text>
    </Tag>
  );
};
