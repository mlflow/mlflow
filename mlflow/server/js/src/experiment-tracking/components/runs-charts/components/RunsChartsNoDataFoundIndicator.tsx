import { Empty, NoIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { useUpdateExperimentViewUIState } from '../../experiment-page/contexts/ExperimentPageUIStateContext';
import { useCallback } from 'react';
import { FormattedMessage } from 'react-intl';

export const RunsChartsNoDataFoundIndicator = () => {
  const updateUIState = useUpdateExperimentViewUIState();
  const { theme } = useDesignSystemTheme();

  const hideEmptyCharts = useCallback(() => {
    updateUIState((state) => ({ ...state, hideEmptyCharts: true }));
  }, [updateUIState]);

  return (
    <div
      css={{
        flex: 1,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        paddingLeft: theme.spacing.lg,
        paddingRight: theme.spacing.lg,
      }}
    >
      <Empty
        description={
          <FormattedMessage
            defaultMessage="No chart data available for the currently visible runs. Select other runs or <link>hide empty charts.</link>"
            description="Experiment tracking > runs charts > indication displayed when no corresponding data is found to be used in chart-based run comparison"
            values={{
              link: (chunks) => (
                <Typography.Link
                  componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_runschartsnodatafoundindicator.tsx_31"
                  onClick={hideEmptyCharts}
                >
                  {chunks}
                </Typography.Link>
              ),
            }}
          />
        }
        image={<NoIcon />}
      />
    </div>
  );
};
