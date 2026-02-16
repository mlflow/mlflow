import { Empty, Typography } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

export const shouldDisplayEvaluationArtifactEmptyState = ({
  noEvalTablesLogged,
  userDeselectedAllColumns,
  areRunsSelected,
  areTablesSelected,
}: EvaluationArtifactViewEmptyStateProps) =>
  !areTablesSelected || !areRunsSelected || userDeselectedAllColumns || noEvalTablesLogged;

interface EvaluationArtifactViewEmptyStateProps {
  noEvalTablesLogged: boolean;
  userDeselectedAllColumns: boolean;
  areRunsSelected: boolean;
  areTablesSelected: boolean;
}

export const EvaluationArtifactViewEmptyState = ({
  noEvalTablesLogged,
  userDeselectedAllColumns,
  areRunsSelected,
}: EvaluationArtifactViewEmptyStateProps) => {
  const getEmptyContent = () => {
    if (!areRunsSelected) {
      return [
        // eslint-disable-next-line react/jsx-key
        <FormattedMessage
          defaultMessage="No runs selected"
          description="Experiment page > artifact compare view > empty state for no runs selected > title"
        />,
        // eslint-disable-next-line react/jsx-key
        <FormattedMessage
          defaultMessage="Make sure that at least one experiment run is visible and available to compare"
          description="Experiment page > artifact compare view > empty state for no runs selected > subtitle with the hint"
        />,
      ];
    }
    if (noEvalTablesLogged) {
      return [
        // eslint-disable-next-line react/jsx-key
        <FormattedMessage
          defaultMessage="No evaluation tables logged"
          description="Experiment page > artifact compare view > empty state for no evaluation tables logged > title"
        />,
        // eslint-disable-next-line react/jsx-key
        <FormattedMessage
          defaultMessage="Please log at least one table artifact containing evaluation data. <link>Learn more</link>."
          description="Experiment page > artifact compare view > empty state for no evaluation tables logged > subtitle"
          values={{
            link: (chunks) => (
              <Typography.Link
                componentId="codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_evaluationartifactviewemptystate.tsx_48"
                openInNewTab
                href="https://mlflow.org/docs/latest/python_api/mlflow.html?highlight=log_table#mlflow.log_table"
                target="_blank"
                rel="noopener noreferrer"
              >
                {chunks}
              </Typography.Link>
            ),
          }}
        />,
      ];
    }
    if (userDeselectedAllColumns) {
      return [
        // eslint-disable-next-line react/jsx-key
        <FormattedMessage
          defaultMessage="No group by columns selected"
          description="Experiment page > artifact compare view > empty state for no group by columns selected > title"
        />,
        // eslint-disable-next-line react/jsx-key
        <FormattedMessage
          defaultMessage='Using controls above, select at least one "group by" column.'
          description="Experiment page > artifact compare view > empty state for no group by columns selected > title"
        />,
      ];
    }
    return [
      // eslint-disable-next-line react/jsx-key
      <FormattedMessage
        defaultMessage="No tables selected"
        description="Experiment page > artifact compare view > empty state for no tables selected > title"
      />,
      // eslint-disable-next-line react/jsx-key
      <FormattedMessage
        defaultMessage="Using controls above, select at least one artifact containing table."
        description="Experiment page > artifact compare view > empty state for no tables selected > subtitle with the hint"
      />,
    ];
  };
  const [title, description] = getEmptyContent();
  return <Empty title={title} description={description} />;
};
