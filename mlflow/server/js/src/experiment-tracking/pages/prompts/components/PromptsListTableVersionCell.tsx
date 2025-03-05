import { Typography } from '@databricks/design-system';
import { ColumnDef } from '@tanstack/react-table';
import { FormattedMessage } from 'react-intl';

export const PromptsListTableVersionCell: ColumnDef<any>['cell'] = ({ row: { original }, getValue }) => {
  const version = getValue<string>();

  if (!version) {
    return null;
  }
  return (
    <Typography.Text>
      <FormattedMessage
        defaultMessage="Version {version}"
        description="TODO"
        values={{
          version,
        }}
      />
    </Typography.Text>
  );
};
