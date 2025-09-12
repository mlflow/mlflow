import { Typography } from '@databricks/design-system';
import type { ColumnDef } from '@tanstack/react-table';
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
        description="Label for the version of a registered prompt in the registered prompts table"
        values={{
          version,
        }}
      />
    </Typography.Text>
  );
};
