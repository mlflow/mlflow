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
import { flexRender, getCoreRowModel, useReactTable } from '@tanstack/react-table';
import { useMemo } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import Utils from '../../../../common/utils/Utils';
import { ModelVersionTableAliasesCell } from '../../../../model-registry/components/aliases/ModelVersionTableAliasesCell';
import type { RegisteredPrompt, RegisteredPromptVersion } from '../types';
import { PromptVersionsTableMode } from '../utils';
import { PromptsListTableVersionCell } from './PromptsListTableVersionCell';
import { PromptVersionsTableAliasesCell } from './PromptVersionsTableAliasesCell';
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
        cell: PromptsListTableVersionCell,
      },
    ];

    if (mode === PromptVersionsTableMode.TABLE) {
      resultColumns.push({
        id: 'creation_timestamp',
        header: intl.formatMessage({
          defaultMessage: 'Registered at',
          description: 'Header for the registration time column in the registered prompts table',
        }),
        accessorFn: ({ creation_timestamp }) => Utils.formatTimestamp(creation_timestamp, intl),
      });

      resultColumns.push({
        id: 'commit_message',
        header: intl.formatMessage({
          defaultMessage: 'Commit message',
          description: 'Header for the commit message column in the registered prompts table',
        }),
        accessorKey: 'description',
      });
      resultColumns.push({
        id: 'aliases',
        header: intl.formatMessage({
          defaultMessage: 'Aliases',
          description: 'Header for the aliases column in the registered prompts table',
        }),
        accessorKey: 'aliases',
        cell: PromptVersionsTableAliasesCell,
      });
    }

    return resultColumns;
  }, [mode, intl]);

  const table = useReactTable({
    data: promptVersions ?? [],
    getRowId: (row) => row.version,
    columns,
    getCoreRowModel: getCoreRowModel(),
    meta: { showEditAliasesModal, aliasesByVersion, registeredPrompt },
  });

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
              [PromptVersionsTableMode.PREVIEW].includes(mode) && selectedVersion === row.original.version;

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

            const showCursorForEntireRow = mode === PromptVersionsTableMode.PREVIEW;
            return (
              <TableRow
                key={row.id}
                css={{
                  height: theme.general.heightBase,
                  backgroundColor: getColor(),
                  cursor: showCursorForEntireRow ? 'pointer' : 'default',
                }}
                onClick={() => {
                  if (mode !== PromptVersionsTableMode.PREVIEW) {
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
