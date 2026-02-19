import type { ColumnDef, ColumnDefTemplate, CellContext } from '@tanstack/react-table';
import { getCoreRowModel, getFilteredRowModel, flexRender } from '@tanstack/react-table';
import { useMemo, useState } from 'react';

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
import { useIntl, FormattedMessage } from '@databricks/i18n';
import { useReactTable_unverifiedWithReact18 as useReactTable } from '@databricks/web-shared/react-table';
import { CodeSnippet } from '@databricks/web-shared/snippet';

import { Link, generatePath } from '../RoutingUtils';

const PROMPT_VERSION_QUERY_PARAM = 'promptVersion';

interface Props {
  data: LinkedPromptsRow[];
}

type LinkedPromptsRow = { experimentId: string; name: string; version: string };

type TableCellRenderer = ColumnDefTemplate<CellContext<LinkedPromptsRow, unknown>>;

const LINK_TRACE_TO_PROMPT_EXAMPLE = `from openai import OpenAI
import mlflow

mlflow.genai.register_prompt("<prompt name>", "What is {{name}}")

@mlflow.trace
def question(name):
    prompt = mlflow.genai.load_prompt("prompts:/<prompt name>@latest")
    client = OpenAI()
    
    messages = [{"role": "user", "content": prompt.format(name=name)}]
    
    response = client.chat.completions.create(
       model="gpt-5-mini",
       messages=messages,
    )
    return response.choices[0].message.content

question("MLflow")`;

const PromptNameCellRenderer: ColumnDef<LinkedPromptsRow>['cell'] = ({ row }) => {
  const { experimentId, name, version } = row.original ?? {};

  const baseRoute = generatePath('/ml/experiments/:experimentId/prompts/:promptName', {
    experimentId,
    promptName: name,
  });

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

export const ModelTraceExplorerLinkedPromptsTable = ({ data }: Props) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const [globalFilter, setGlobalFilter] = useState('');

  const renderPromptLinkingCard = () => (
    <div
      css={{
        borderRadius: theme.general.borderRadiusBase,
        border: `1px solid ${theme.colors.border}`,
        backgroundColor: theme.colors.backgroundSecondary,
        padding: theme.spacing.lg,
        width: '100%',
      }}
    >
      <Typography.Text bold>
        <FormattedMessage
          defaultMessage="Link prompts to traces"
          description="Heading describing how to add prompts when no prompt records are available"
        />
      </Typography.Text>
      <Typography.Paragraph css={{ marginTop: theme.spacing.sm }}>
        <FormattedMessage
          defaultMessage="Link prompts to your traces by loading your registered prompts inside the traced function."
          description="Helper text describing how to link prompts when no prompt records are available"
        />
      </Typography.Paragraph>
      <CodeSnippet language="python" showLineNumbers wrapLongLines theme={theme.isDarkMode ? 'duotoneDark' : 'light'}>
        {LINK_TRACE_TO_PROMPT_EXAMPLE}
      </CodeSnippet>
    </div>
  );

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

  const table = useReactTable(
    'js/packages/web-shared/src/model-trace-explorer/linked-prompts/ModelTraceExplorerLinkedPromptsTable.tsx',
    {
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
    },
  );

  const renderTableContent = () => {
    return (
      <>
        <div css={{ marginBottom: theme.spacing.sm }}>
          <Input
            componentId="shared.model-trace-explorer.linked_prompts.table.search"
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
              <div
                css={{
                  gap: theme.spacing.lg,
                }}
              >
                {renderPromptLinkingCard()}
              </div>
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
                componentId="shared.model-trace-explorer.linked_prompts.table.header"
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
    <div
      css={{
        padding: theme.spacing.md,
        flex: 1,
        borderRadius: theme.general.borderRadiusBase,
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
      }}
    >
      {renderTableContent()}
    </div>
  );
};
