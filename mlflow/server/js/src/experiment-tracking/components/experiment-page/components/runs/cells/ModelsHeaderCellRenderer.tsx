import React from 'react';
import {
  SortAscendingIcon,
  SortDescendingIcon,
  LegacyTooltip,
  useDesignSystemTheme,
  InfoTooltip,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { ATTRIBUTE_COLUMN_LABELS } from '../../../../../constants';
import {
  shouldUnifyLoggedModelsAndRegisteredModels,
  shouldUseGetLoggedModelsBatchAPI,
} from '../../../../../../common/utils/FeatureUtils';

export const ModelsHeaderCellRenderer = React.memo(() => {
  const { theme } = useDesignSystemTheme();

  // Check if we are using:
  // - unified (registered and logged) models
  // - models based on run's inputs and outputs
  // We'll use it to display better tooltip
  const isUsingUnifiedModels = shouldUnifyLoggedModelsAndRegisteredModels() && shouldUseGetLoggedModelsBatchAPI();

  return (
    <div
      role="columnheader"
      css={{
        height: '100%',
        width: '100%',
        display: 'flex',
        alignItems: 'center',
        padding: '0 12px',
        gap: theme.spacing.xs,
      }}
    >
      {isUsingUnifiedModels ? (
        <>
          {ATTRIBUTE_COLUMN_LABELS.MODELS}
          <InfoTooltip
            componentId="mlflow.experiment_view_runs_table.column_header.models.tooltip"
            content={
              <FormattedMessage
                defaultMessage="This column contains all models logged or evaluated by the run. Click into an individual run to see more detailed information about all models associated with it."
                description='A descriptive tooltip for the "Models" column header in the runs table on the MLflow experiment detail page'
              />
            }
          />
        </>
      ) : (
        <LegacyTooltip
          title={
            <FormattedMessage
              defaultMessage="Click into an individual run to see all models associated with it"
              description='MLflow experiment detail page > runs table > tooltip on ML "Models" column header'
            />
          }
        >
          {ATTRIBUTE_COLUMN_LABELS.MODELS}
        </LegacyTooltip>
      )}
    </div>
  );
});
