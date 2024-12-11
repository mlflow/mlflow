import React from 'react';
import { SortAscendingIcon, SortDescendingIcon, LegacyTooltip, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { ATTRIBUTE_COLUMN_LABELS } from '../../../../../constants';

export const ModelsHeaderCellRenderer = React.memo(() => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      role="columnheader"
      css={{
        height: '100%',
        width: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 12px',
        gap: theme.spacing.sm,
      }}
    >
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
    </div>
  );
});
