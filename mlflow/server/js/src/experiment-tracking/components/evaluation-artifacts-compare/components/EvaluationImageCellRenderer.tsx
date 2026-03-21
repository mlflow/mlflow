import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import type { ICellRendererParams } from '@ag-grid-community/core';
import { FormattedMessage } from 'react-intl';
import type { RunRowType } from '../../experiment-page/utils/experimentPage.row-types';
import type { UseEvaluationArtifactTableDataResult } from '../hooks/useEvaluationArtifactTableData';
import { ImagePlot } from '@mlflow/mlflow/src/experiment-tracking/components/runs-charts/components/charts/ImageGridPlot.common';
import type { EvaluateCellImage } from '@mlflow/mlflow/src/experiment-tracking/types';

interface EvaluationImageCellRendererProps extends ICellRendererParams {
  value: EvaluateCellImage;
  isGroupByColumn?: boolean;
  context: { highlightedText: string };

  data: UseEvaluationArtifactTableDataResult extends (infer U)[] ? U : UseEvaluationArtifactTableDataResult;

  // Valid only for run columns
  run?: RunRowType;
}

/**
 * Component used to render a single text cell in the evaluation artifacts comparison table.
 */
/* eslint-disable complexity */
export const EvaluationImageCellRenderer = ({ value }: EvaluationImageCellRendererProps) => {
  const { theme } = useDesignSystemTheme();

  const backgroundColor = theme.colors.backgroundPrimary;

  return (
    <div
      css={{
        height: '100%',
        whiteSpace: 'normal',
        padding: theme.spacing.sm,
        overflow: 'hidden',
        position: 'relative',
        cursor: 'pointer',
        backgroundColor,
        '&:hover': {
          backgroundColor: theme.colors.actionDefaultBackgroundHover,
        },
      }}
    >
      {!value || !value.url || !value.compressed_url ? (
        <Typography.Text color="info" css={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>
          <FormattedMessage
            defaultMessage="(empty)"
            description="Experiment page > artifact compare view > results table > no result (empty cell)"
          />
        </Typography.Text>
      ) : (
        <span
          css={{
            display: '-webkit-box',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            WebkitBoxOrient: 'vertical',
            WebkitLineClamp: '7',
            width: '100%',
            height: '100%',
          }}
        >
          <ImagePlot imageUrl={value.url} compressedImageUrl={value.compressed_url} />
        </span>
      )}
    </div>
  );
};
