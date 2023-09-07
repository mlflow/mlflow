import { Button, VisibleIcon, useDesignSystemTheme } from '@databricks/design-system';
import { Link } from 'react-router-dom-v5-compat';
import ExperimentRoutes from '../../../routes';
import { RunRowType } from '../../experiment-page/utils/experimentPage.row-types';
import {
  EVALUATION_ARTIFACTS_RUN_NAME_HEIGHT,
  getEvaluationArtifactsTableHeaderHeight,
} from '../EvaluationArtifactCompare.utils';
import { EvaluationRunHeaderModelIndicator } from './EvaluationRunHeaderModelIndicator';
import { shouldEnableExperimentDatasetTracking } from '../../../../common/utils/FeatureUtils';
import { EvaluationRunHeaderDatasetIndicator } from './EvaluationRunHeaderDatasetIndicator';
import type { RunDatasetWithTags } from '../../../types';

interface EvaluationRunHeaderCellRendererProps {
  run: RunRowType;
  onHideRun: (runUuid: string) => void;
  onDatasetSelected: (dataset: RunDatasetWithTags, run: RunRowType) => void;
}

/**
 * Component used as a column header for output ("run") columns
 */
export const EvaluationRunHeaderCellRenderer = ({
  run,
  onHideRun,
  onDatasetSelected,
}: EvaluationRunHeaderCellRendererProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        height: getEvaluationArtifactsTableHeaderHeight(),
        width: '100%',
        display: 'flex',
        flexDirection: 'column',
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
          borderTop: `1px solid ${theme.colors.borderDecorative}`,
          borderBottom: `1px solid ${theme.colors.borderDecorative}`,
        }}
      >
        <Link
          css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'center' }}
          to={ExperimentRoutes.getRunPageRoute(run.experimentId, run.runUuid)}
          target='_blank'
        >
          <div
            css={{
              backgroundColor: run.color,
              width: 12,
              height: 12,
              borderRadius: 6,
            }}
          />

          {run.runName}
        </Link>
        <div css={{ flexBasis: theme.spacing.sm }} />
        <Button onClick={() => onHideRun(run.runUuid)} size='small' icon={<VisibleIcon />} />
      </div>
      <div
        css={{
          flex: 1,
          paddingTop: theme.spacing.xs,
          paddingBottom: theme.spacing.xs,
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        {shouldEnableExperimentDatasetTracking() && (
          <EvaluationRunHeaderDatasetIndicator run={run} onDatasetSelected={onDatasetSelected} />
        )}
        <EvaluationRunHeaderModelIndicator run={run} />
      </div>
    </div>
  );
};
