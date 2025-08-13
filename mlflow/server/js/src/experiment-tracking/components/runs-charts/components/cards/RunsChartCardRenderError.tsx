import { DangerIcon, Empty } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

export const RunsChartCardRenderError = () => (
  <div css={{ flex: 1, height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
    <Empty
      title={
        <FormattedMessage
          defaultMessage="Failed to load chart"
          description="Title for the error message when the MLflow chart fails to load"
        />
      }
      description={
        <FormattedMessage
          defaultMessage="There was an unrecoverable error while loading the chart. Please try to reconfigure the chart and/or reload the window."
          description="Description for the error message when the MLflow chart fails to load"
        />
      }
      image={<DangerIcon />}
    />
  </div>
);
