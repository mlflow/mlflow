import type { ColumnDef } from '@tanstack/react-table';
import { Link } from '../../../../common/utils/RoutingUtils';
import Routes from '../../../routes';
import type { RegisteredPrompt } from '../types';
import type { PromptsTableMetadata } from '../utils';

export const PromptsListTableNameCell: ColumnDef<RegisteredPrompt>['cell'] = ({
  row: { original },
  getValue,
  table: {
    options: { meta },
  },
}) => {
  const name = getValue<string>();
  const { experimentId } = (meta || {}) as PromptsTableMetadata;

  if (!original.name) {
    return name;
  }
  return <Link to={Routes.getPromptDetailsPageRoute(encodeURIComponent(original.name), experimentId)}>{name}</Link>;
};
