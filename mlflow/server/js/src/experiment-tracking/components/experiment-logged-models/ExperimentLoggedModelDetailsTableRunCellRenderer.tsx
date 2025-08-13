import type { CellContext, ColumnDefTemplate } from '@tanstack/react-table';
import { Link } from '../../../common/utils/RoutingUtils';
import Routes from '../../routes';

export const ExperimentLoggedModelDetailsTableRunCellRenderer: ColumnDefTemplate<
  CellContext<
    unknown,
    {
      runId?: string | null;
      runName?: string | null;
      experimentId?: string | null;
    }
  >
> = ({ getValue }) => {
  const { runName, runId } = getValue() ?? {};

  return <Link to={Routes.getDirectRunPageRoute(runId ?? '')}>{runName || runId}</Link>;
};
