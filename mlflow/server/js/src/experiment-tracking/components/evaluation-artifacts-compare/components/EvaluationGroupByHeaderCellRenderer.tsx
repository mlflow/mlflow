import { Tooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';
import {
  EVALUATION_ARTIFACTS_RUN_NAME_HEIGHT,
  getEvaluationArtifactsTableHeaderHeight,
} from '../EvaluationArtifactCompare.utils';
import { truncate } from 'lodash';

interface EvaluationGroupByHeaderCellRendererProps {
  displayName: string;
}

/**
 * Component used as a column header for "group by" columns
 */
export const EvaluationGroupByHeaderCellRenderer = ({
  displayName,
}: EvaluationGroupByHeaderCellRendererProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        height: getEvaluationArtifactsTableHeaderHeight(),
        width: '100%',
        borderBottom: `1px solid ${theme.colors.borderDecorative}`,
      }}
    >
      <div
        css={{
          height: EVALUATION_ARTIFACTS_RUN_NAME_HEIGHT,
          width: '100%',
          display: 'flex',
          padding: theme.spacing.sm,
          alignItems: 'center',
          border: `1px solid ${theme.colors.borderDecorative}`,
          borderLeft: 'none',
          overflow: 'hidden',
          whiteSpace: 'nowrap',
        }}
      >
        <Tooltip title={truncate(displayName, { length: 250 })}>
          <Typography.Text bold css={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>
            {displayName}
          </Typography.Text>
        </Tooltip>
      </div>
    </div>
  );
};
