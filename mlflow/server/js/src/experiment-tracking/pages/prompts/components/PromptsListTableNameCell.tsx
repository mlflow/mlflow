import type { ColumnDef } from '@tanstack/react-table';
import { Link } from '../../../../common/utils/RoutingUtils';
import Routes from '../../../routes';
import type { RegisteredPrompt } from '../types';

export const PromptsListTableNameCell: ColumnDef<RegisteredPrompt>['cell'] = ({ row: { original }, getValue }) => {
  const name = getValue<string>();

  if (!original.name) {
    return name;
  }
  return <Link to={Routes.getPromptDetailsPageRoute(encodeURIComponent(original.name))}>{name}</Link>;
};
