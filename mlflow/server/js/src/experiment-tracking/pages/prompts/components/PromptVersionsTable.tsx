import { useReactTable_unverifiedWithReact18 as useReactTable } from '@databricks/web-shared/react-table';
import {
  ChevronRightIcon,
  Empty,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  TableSkeletonRows,
  useDesignSystemTheme,
} from '@databricks/design-system';
import type { ColumnDef } from '@tanstack/react-table';
import { flexRender, getCoreRowModel } from '@tanstack/react-table';
import { useMemo } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import type { RegisteredPrompt, RegisteredPromptVersion } from '../types';
import { PromptVersionsTableMode } from '../utils';
import { PromptVersionsTableCombinedCell } from './PromptVersionsTableCombinedCell';
import { PromptVersionsDiffSelectorButton } from './PromptVersionsDiffSelectorButton';

type PromptVersionsTableColumnDef = ColumnDef<RegisteredPromptVersion>;

export const PromptVersionsTable = ({
  promptVersions,
  onUpdateComparedVersion,
  isLoading,
  onUpdateSelectedVersion,
  comparedVersion,
  selectedVersion,
  mode,
  registeredPrompt,
  showEditAliasesModal,
  aliasesByVersion,
}: {
  promptVersions?: RegisteredPromptVersion[];
  isLoading: boolean;
  selectedVersion?: string;
  comparedVersion?: string;
  onUpdateSelectedVersion: (version: string) => void;
  onUpdateComparedVersion: (version: string) => void;
  mode: PromptVersionsTableMode;
  registeredPrompt?: RegisteredPrompt;
  showEditAliasesModal?: (versionNumber: string) => void;
  aliasesByVersion: Record<string, string[]>;
}) => {
  const intl = useIntl();

  const { theme } = useDesignSystemTheme();
  const columns = useMemo(() => {
    const resultColumns: PromptVersionsTableColumnDef[] = [
      {
        id: 'version',
        header: intl.formatMessage({
          defaultMessage: 'Version',
          description: 'Header for the version column in the registered prompts table',
        }),
        accessorKey: 'version',
        cell: PromptVersionsTableCombinedCell,
      },
    ];

    return resultColumns;
  }, [intl]);

  const table = useReactTable(
    'mlflow/server/js/src/experiment-tracking/pages/prompts/components/PromptVersionsTable.tsx',
    {
      data: promptVersions ?? [],
      getRowId: (row) => row.version,
      columns,
      getCoreRowModel: getCoreRowModel(),
      meta: { showEditAliasesModal, aliasesByVersion, registeredPrompt },
    },
  );

  const getEmptyState = () => {
    if (!isLoading && promptVersions?.length === 0) {
      return (
        <Empty
          title={
            <FormattedMessage
              defaultMessage="No prompt versions created"
              description="A header for the empty state in the prompt versions table"
            />
          }
          description={
            <FormattedMessage
              defaultMessage='Use "Create prompt version" button in order to create a new prompt version'
              description="Guidelines for the user on how to create a new prompt version in the prompt versions table"
            />
          }
        />
      );
    }

    return null;
  };

  return (
    <div css={{ flex: 1, overflow: 'hidden' }}>
      <Table scrollable empty={getEmptyState()} aria-label="Prompt versions table">
        <TableRow isHeader>
          {table.getLeafHeaders().map((header) => (
            <TableHeader componentId="mlflow.prompts.versions.table.header" key={header.id}>
              {flexRender(header.column.columnDef.header, header.getContext())}
            </TableHeader>
          ))}
        </TableRow>
        {isLoading ? (
          <TableSkeletonRows table={table} />
        ) : (
          table.getRowModel().rows.map((row) => {
            const isSelectedSingle =
              [PromptVersionsTableMode.PREVIEW, PromptVersionsTableMode.TRACES].includes(mode) &&
              selectedVersion === row.original.version;

            const isSelectedFirstToCompare =
              [PromptVersionsTableMode.COMPARE].includes(mode) && selectedVersion === row.original.version;

            const isSelectedSecondToCompare =
              [PromptVersionsTableMode.COMPARE].includes(mode) && comparedVersion === row.original.version;

            const getColor = () => {
              if (isSelectedSingle) {
                return theme.colors.actionDefaultBackgroundPress;
              } else if (isSelectedFirstToCompare) {
                return theme.colors.actionDefaultBackgroundHover;
              } else if (isSelectedSecondToCompare) {
                return theme.colors.actionDefaultBackgroundHover;
              }
              return 'transparent';
            };

            const showCursorForEntireRow =
              mode === PromptVersionsTableMode.PREVIEW || mode === PromptVersionsTableMode.TRACES;
            return (
              <TableRow
                key={row.id}
                css={{
                  backgroundColor: getColor(),
                  cursor: showCursorForEntireRow ? 'pointer' : 'default',
                }}
                onClick={() => {
                  if (mode !== PromptVersionsTableMode.PREVIEW && mode !== PromptVersionsTableMode.TRACES) {
                    return;
                  }
                  onUpdateSelectedVersion(row.original.version);
                }}
              >
                {row.getAllCells().map((cell) => (
                  <TableCell key={cell.id} css={{ alignItems: 'center' }}>
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </TableCell>
                ))}
                {isSelectedSingle && (
                  <div
                    css={{
                      width: theme.spacing.md * 2,
                      display: 'flex',
                      alignItems: 'center',
                      paddingRight: theme.spacing.sm,
                    }}
                  >
                    <ChevronRightIcon />
                  </div>
                )}
                {mode === PromptVersionsTableMode.COMPARE && (
                  <PromptVersionsDiffSelectorButton
                    onSelectFirst={() => onUpdateSelectedVersion(row.original.version)}
                    onSelectSecond={() => onUpdateComparedVersion(row.original.version)}
                    isSelectedFirstToCompare={isSelectedFirstToCompare}
                    isSelectedSecondToCompare={isSelectedSecondToCompare}
                  />
                )}
              </TableRow>
            );
          })
        )}
      </Table>
    </div>
  );
};
