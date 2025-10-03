import { Typography } from '@databricks/design-system';
import { useDesignSystemTheme } from '@databricks/design-system';
import { Row } from '@tanstack/react-table';
import { EvaluationDataset } from '../types';

export const NameCell = ({ row }: { row: Row<EvaluationDataset> }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div css={{ overflow: 'hidden', display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
      <Typography.Link
        css={{ textOverflow: 'ellipsis', whiteSpace: 'nowrap', overflow: 'hidden', flexShrink: 1 }}
        componentId="mlflow.eval-datasets.dataset-name-cell"
      >
        {row.original.name}
      </Typography.Link>
    </div>
  );
};
