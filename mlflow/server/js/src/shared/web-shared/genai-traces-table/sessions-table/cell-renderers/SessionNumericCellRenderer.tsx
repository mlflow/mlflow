import { Tag } from '@databricks/design-system';
import { CellContext } from '@tanstack/react-table';
import { SessionTableRow } from '../types';

export const SessionNumericCellRenderer = (props: CellContext<SessionTableRow, unknown>) => {
  const { cell } = props;
  const value = cell.getValue();

  return <Tag componentId="mlflow.genai-traces-table.session-tokens">{value}</Tag>;
};
