import { useDesignSystemTheme } from '@databricks/design-system';
import { Link } from '../../../common/utils/RoutingUtils';
import Routes from '../../routes';
import { LoggedModelProto } from '../../types';
import { getStableColorForRun } from '../../utils/RunNameUtils';
import { RunColorPill } from '../experiment-page/components/RunColorPill';

export const ExperimentLoggedModelTableNameCell = ({ data }: { data: LoggedModelProto }) => {
  const { theme } = useDesignSystemTheme();
  if (!data.info?.experiment_id || !data.info?.model_id) {
    return <>{data.info?.name}</>;
  }
  return (
    <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
      {/* TODO: consider how we should determine the color of the model */}
      <RunColorPill color={getStableColorForRun(data.info.model_id)} />
      <Link to={Routes.getExperimentLoggedModelDetailsPageRoute(data.info.experiment_id, data.info.model_id)}>
        {data.info.name}
      </Link>
    </div>
  );
};
