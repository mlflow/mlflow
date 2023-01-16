import React from 'react';
import { Link } from 'react-router-dom';
import Routes from '../../../../../routes';
import { RunRowType } from '../../../utils/experimentPage.row-types';

export interface ExperimentNameCellRendererProps {
  value: {
    name: string;
    basename: string;
  };
  data: RunRowType;
}

export const ExperimentNameCellRenderer = React.memo(
  ({ data, value }: ExperimentNameCellRendererProps) => (
    <Link to={Routes.getExperimentPageRoute(data.experimentId)} title={value.name}>
      {value.basename}
    </Link>
  ),
);
