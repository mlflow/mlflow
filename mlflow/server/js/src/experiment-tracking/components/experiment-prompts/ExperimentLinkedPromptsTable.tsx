import { useMemo, useState } from 'react';
import { useIntl } from 'react-intl';

import {
  useDesignSystemTheme,
  Typography,
  Table,
  Empty,
  TableRow,
  TableHeader,
  TableCell,
  Input,
  getShadowScrollStyles,
  SearchIcon,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import type { ColumnDef, ColumnDefTemplate, CellContext } from '@tanstack/react-table';
import { useReactTable, getCoreRowModel, getFilteredRowModel, flexRender } from '@tanstack/react-table';

import { Link } from '../../../common/utils/RoutingUtils';
import Routes from '../../routes';
import { PROMPT_VERSION_QUERY_PARAM } from '../../pages/prompts/utils';

interface Props {
  data: LinkedPromptsRow[];
}

type LinkedPromptsRow = { experimentId: string; name: string; version: string };

type TableCellRenderer = ColumnDefTemplate<CellContext<LinkedPromptsRow, unknown>>;

const PromptNameCellRenderer: ColumnDef<LinkedPromptsRow>['cell'] = ({ row }) => {
  const { experimentId, name, version } = row.original ?? {};

  // TODO: allow linking to prompt versions in OSS
  const baseRoute = Routes.getPromptDetailsPageRoute(name);

  if (version) {
    const searchParams = new URLSearchParams();
    searchParams.set(PROMPT_VERSION_QUERY_PARAM, version);
    const routeWithVersion = `${baseRoute}?${searchParams.toString()}`;
    return <Link to={routeWithVersion}>{name}</Link>;
  }

  return <Link to={baseRoute}>{name}</Link>;
};

const VersionCellRenderer: ColumnDef<LinkedPromptsRow>['cell'] = ({ row }) => {
  const { version } = row.original ?? {};

  return <Typography.Paragraph withoutMargins>{version}</Typography.Paragraph>;
};

export const ExperimentLinkedPromptsTable = ({ data }: Props) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const [globalFilter, setGlobalFilter] = useState('');

  const columns = useMemo<ColumnDef<any>[]>(
    () => [
      {
        id: 'name',
        header: intl.formatMessage({
          defaultMessage: 'Prompt Name',
          description: 'Header for prompt name column in linked prompts table on logged model details page',
        }),
        enableResizing: true,
        size: 400,
        accessorKey: 'name',
        cell: PromptNameCellRenderer as TableCellRenderer,
      },
      {
        id: 'version',
        header: intl.formatMessage({
          defaultMessage: 'Version',
          description: 'Header for version column in linked prompts table on logged model details page',
        }),
        enableResizing: false,
        accessorKey: 'version',
        cell: VersionCellRenderer as TableCellRenderer,
      },
    ],
    [intl],
  );

  const table = useReactTable({
    data,
    getRowId: (row) => row.name,
    getCoreRowModel: getCoreRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    enableColumnResizing: true,
    columnResizeMode: 'onChange',
    columns,
    state: {
      globalFilter,
    },
    onGlobalFilterChange: setGlobalFilter,
    globalFilterFn: 'includesString',
  });

  const renderTableContent = () => {
    return (
      <>
        <div css={{ marginBottom: theme.spacing.sm }}>
          <Input
            componentId="mlflow.logged_model.details.runs.table.search"
            prefix={<SearchIcon />}
            placeholder={intl.formatMessage({
              defaultMessage: 'Search prompts',
              description:
                'Placeholder text for the search input in the prompts table on the logged model details page',
            })}
            value={globalFilter}
            onChange={(e) => setGlobalFilter(e.target.value)}
            allowClear
          />
        </div>
        <Table
          scrollable
          css={{
            '&>div': getShadowScrollStyles(theme, {
              orientation: 'vertical',
            }),
          }}
          empty={
            data.length === 0 ? (
              <Empty
                description={
                  <FormattedMessage
                    defaultMessage="No prompts"
                    description="No results message for linked prompts table on logged model details page"
                  />
                }
              />
            ) : table.getFilteredRowModel().rows.length === 0 ? (
              <Empty
                description={
                  <FormattedMessage
                    defaultMessage="No prompts match your search"
                    description="No search results message for linked prompts table on logged model details page"
                  />
                }
              />
            ) : null
          }
        >
          <TableRow isHeader>
            {table.getLeafHeaders().map((header) => (
              <TableHeader
                componentId="mlflow.logged_model.details.linked_prompts.table.header"
                key={header.id}
                header={header}
                column={header.column}
                setColumnSizing={table.setColumnSizing}
                isResizing={header.column.getIsResizing()}
                css={{
                  flexGrow: header.column.getCanResize() ? 0 : 1,
                }}
                style={{
                  flexBasis: header.column.getCanResize() ? header.column.getSize() : undefined,
                }}
              >
                {flexRender(header.column.columnDef.header, header.getContext())}
              </TableHeader>
            ))}
          </TableRow>
          {table.getRowModel().rows.map((row) => (
            <TableRow key={`${row.id}-${row.original.name}-${row.original.version}-row`}>
              {row.getAllCells().map((cell) => (
                <TableCell
                  key={`${cell.id}-${row.original.name}-${row.original.version}-cell`}
                  style={{
                    flexGrow: cell.column.getCanResize() ? 0 : 1,
                    flexBasis: cell.column.getCanResize() ? cell.column.getSize() : undefined,
                  }}
                >
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </Table>
      </>
    );
  };

  return (
    <div css={{ display: 'flex', flexDirection: 'column', overflow: 'hidden', maxHeight: 400 }}>
      <Typography.Title css={{ fontSize: 16 }}>
        <FormattedMessage
          defaultMessage="Prompts"
          description="Title for linked prompts table on logged model details page"
        />
      </Typography.Title>
      <div
        css={{
          padding: theme.spacing.sm,
          border: `1px solid ${theme.colors.border}`,
          borderRadius: theme.general.borderRadiusBase,
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
      >
        {renderTableContent()}
      </div>
    </div>
  );
};
