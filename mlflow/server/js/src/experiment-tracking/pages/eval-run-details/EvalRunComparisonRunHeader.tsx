import { FormattedMessage } from 'react-intl';
import { Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { RunColorPill } from '../../components/experiment-page/components/RunColorPill';
import { useGetExperimentRunColor } from '../../components/experiment-page/hooks/useExperimentRunColor';
import { Link } from '../../../common/utils/RoutingUtils';
import Routes from '../../routes';

export interface EvalRunComparisonRunHeaderProps {
  experimentId: string;
  runUuid: string;
  runName: string;
  isBaseline: boolean;
}

export const EvalRunComparisonRunHeader = ({
  experimentId,
  runUuid,
  runName,
  isBaseline,
}: EvalRunComparisonRunHeaderProps) => {
  const { theme } = useDesignSystemTheme();
  const getRunColor = useGetExperimentRunColor();
  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.xs,
        minWidth: 0,
      }}
    >
      <RunColorPill color={getRunColor(runUuid)} />
      <Link
        componentId="mlflow.eval_run_details.compare.run_link"
        to={Routes.getEvaluationRunDetailsRoute(experimentId, runUuid)}
        css={{
          minWidth: 0,
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
        }}
        title={runName}
      >
        <Typography.Text bold>{runName}</Typography.Text>
      </Link>
      {isBaseline && (
        <Tag componentId="mlflow.eval_run_details.compare.baseline" color="turquoise">
          <FormattedMessage
            defaultMessage="Baseline"
            description="Badge for the baseline run in evaluation comparison tables"
          />
        </Tag>
      )}
    </div>
  );
};
