import { Table, TableCell, TableHeader, TableRow } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

export interface ExperimentViewDatasetSchemaTableProps {
  schema: any[];
  filter: string;
}

export const ExperimentViewDatasetSchemaTable = ({
  schema,
  filter,
}: ExperimentViewDatasetSchemaTableProps): JSX.Element => {
  const hasFilter = (name?: string, type?: string) => {
    return (
      filter === '' ||
      name?.toLowerCase().includes(filter.toLowerCase()) ||
      type?.toLowerCase().includes(filter.toLowerCase())
    );
  };

  const filteredSchema = schema.filter((row: { name: string; type: string }, _: number) =>
    hasFilter(row.name, row.type),
  );

  const getNameHeader = () => {
    return (
      <FormattedMessage
        defaultMessage="Name"
        description={'Header for "name" column in the experiment run dataset schema'}
      />
    );
  };

  const getTypeHeader = () => {
    return <FormattedMessage defaultMessage="Type" description={'Header for "type" column in the UC table schema'} />;
  };

  return (
    <Table scrollable css={{ width: '100%' }}>
      <TableRow isHeader>
        <TableHeader>{getNameHeader()}</TableHeader>
        <TableHeader>{getTypeHeader()}</TableHeader>
      </TableRow>
      <div onWheel={(e) => e.stopPropagation()}>
        {filteredSchema.length === 0 ? (
          <TableRow>
            <TableCell>
              <FormattedMessage
                defaultMessage="No results match this search."
                description="No results message in datasets drawer table"
              />
            </TableCell>
          </TableRow>
        ) : (
          filteredSchema.map((row: { name: string; type: string }, idx: number) => (
            <TableRow key={`table-body-row-${idx}`}>
              <TableCell>{row.name}</TableCell>
              <TableCell>{row.type}</TableCell>
            </TableRow>
          ))
        )}
      </div>
    </Table>
  );
};
