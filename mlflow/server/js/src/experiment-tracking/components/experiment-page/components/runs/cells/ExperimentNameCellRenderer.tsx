import React from 'react';
import { Link } from '../../../../../../common/utils/RoutingUtils';
import Routes from '../../../../../routes';
import type { RunRowType } from '../../../utils/experimentPage.row-types';

export interface ExperimentNameCellRendererProps {
  value: {
    name: string;
    basename: string;
  };
  data: RunRowType;
}

// eslint-disable-next-line react-component-name/react-component-name -- TODO(FEINF-4716)
export const ExperimentNameCellRenderer = React.memo(({ data, value }: ExperimentNameCellRendererProps) =>
  !data.experimentId ? null : (
    <Link to={Routes.getExperimentPageRoute(data.experimentId)} title={value.name}>
      {value.basename}
    </Link>
  ),
);
