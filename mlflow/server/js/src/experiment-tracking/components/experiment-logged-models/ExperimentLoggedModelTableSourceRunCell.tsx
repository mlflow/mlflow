import type { GraphQLExperimentRun, LoggedModelProto } from '../../types';
import { Link } from '../../../common/utils/RoutingUtils';
import Routes from '../../routes';

interface LoggedModelWithSourceRun extends LoggedModelProto {
  sourceRun?: GraphQLExperimentRun;
}

export const ExperimentLoggedModelTableSourceRunCell = ({ data }: { data: LoggedModelWithSourceRun }) => {
  if (data.info?.experiment_id && data.info?.source_run_id) {
    return (
      <Link to={Routes.getRunPageRoute(data.info?.experiment_id, data.info?.source_run_id)} target="_blank">
        {data.sourceRun?.info?.runName ?? data.info?.source_run_id}
      </Link>
    );
  }
  return data.info?.source_run_id || <>-</>;
};
