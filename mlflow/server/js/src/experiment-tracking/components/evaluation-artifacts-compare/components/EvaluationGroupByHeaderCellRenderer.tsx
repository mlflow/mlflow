import type { IHeaderParams } from '@ag-grid-community/core';
import { LegacyTooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { truncate } from 'lodash';
import { EvaluationTableHeader } from './EvaluationTableHeader';

interface EvaluationGroupByHeaderCellRendererProps extends IHeaderParams {
  displayName: string;
  onAddNewInputs: () => void;
  displayAddNewInputsButton?: boolean;
}

/**
 * Component used as a column header for "group by" columns
 */
export const EvaluationGroupByHeaderCellRenderer = ({ displayName }: EvaluationGroupByHeaderCellRendererProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <EvaluationTableHeader css={{ justifyContent: 'flex-start', padding: theme.spacing.sm }}>
      <LegacyTooltip title={truncate(displayName, { length: 250 })}>
        <Typography.Text bold css={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>
          {displayName}
        </Typography.Text>
      </LegacyTooltip>
    </EvaluationTableHeader>
  );
};
