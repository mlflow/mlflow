import {
  Button,
  ChevronDownIcon,
  ChevronRightIcon,
  Empty,
  Input,
  SearchIcon,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { KeyValueEntity } from '../../../types';
import { throttle, values } from 'lodash';
import { useEffect, useMemo, useRef, useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { ColumnDef, flexRender, getCoreRowModel, getExpandedRowModel, useReactTable } from '@tanstack/react-table';
import { Interpolation, Theme } from '@emotion/react';

type ParamsColumnDef = ColumnDef<KeyValueEntity> & {
  meta?: { styles?: Interpolation<Theme>; multiline?: boolean };
};

/**
 * Displays cell with expandable parameter value.
 */
const ExpandableParamValueCell = ({
  name,
  value,
  toggleExpanded,
  isExpanded,
  autoExpandedRowsList,
}: {
  name: string;
  value: string;
  toggleExpanded: () => void;
  isExpanded: boolean;
  autoExpandedRowsList: Record<string, boolean>;
}) => {
  const { theme } = useDesignSystemTheme();
  const cellRef = useRef<HTMLDivElement>(null);
  const [isOverflowing, setIsOverflowing] = useState(false);

  useEffect(() => {
    if (autoExpandedRowsList[name]) {
      return;
    }
    if (isOverflowing) {
      toggleExpanded();
      autoExpandedRowsList[name] = true;
    }
  }, [autoExpandedRowsList, isOverflowing, name, toggleExpanded]);

  // Check if cell is overflowing using resize observer
  useEffect(() => {
    if (!cellRef.current) return;

    const resizeObserverCallback: ResizeObserverCallback = throttle(
      ([entry]) => {
        const isOverflowing = entry.target.scrollHeight > entry.target.clientHeight;
        setIsOverflowing(isOverflowing);
      },
      500,
      { trailing: true },
    );

    const resizeObserver = new ResizeObserver(resizeObserverCallback);
    resizeObserver.observe(cellRef.current);
    return () => resizeObserver.disconnect();
  }, [cellRef, toggleExpanded]);

  // Re-check if cell is overflowing after collapse
  useEffect(() => {
    if (!cellRef.current) return;
    if (!isExpanded) {
      const isOverflowing = cellRef.current.scrollHeight > cellRef.current.clientHeight;
      if (isOverflowing) {
        setIsOverflowing(true);
      }
    }
  }, [isExpanded]);

  return (
    <div css={{ display: 'flex', gap: theme.spacing.xs }}>
      {(isOverflowing || isExpanded) && (
        <Button
          componentId="codegen_mlflow_app_src_experiment-tracking_components_run-page_overview_runviewparamstable.tsx_74"
          size="small"
          icon={isExpanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
          onClick={() => toggleExpanded()}
          css={{ flexShrink: 0 }}
        />
      )}
      <div
        title={value}
        css={{
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          display: '-webkit-box',
          '-webkit-box-orient': 'vertical',
          '-webkit-line-clamp': isExpanded ? undefined : '3',
        }}
        ref={cellRef}
      >
        {isExpanded ? <ExpandedParamCell value={value} /> : value}
      </div>
    </div>
  );
};

/**
 * Displays expanded cell with pretty printed JSON.
 */
const ExpandedParamCell = ({ value }: { value: string }) => {
  const structuredJSONValue = useMemo(() => {
    // Attempts to parse the value as JSON and returns a pretty printed version if successful.
    // If JSON structure is not found, returns null.
    try {
      const objectData = JSON.parse(value);
      return JSON.stringify(objectData, null, 2);
    } catch (e) {
      return null;
    }
  }, [value]);
  return (
    <div
      css={{
        whiteSpace: 'pre-wrap',
        wordBreak: 'break-word',
        fontFamily: structuredJSONValue ? 'monospace' : undefined,
      }}
    >
      {structuredJSONValue || value}
    </div>
  );
};

/**
 * Displays table with parameters key/values in run detail overview.
 */
export const RunViewParamsTable = ({
  params,
  runUuid,
}: {
  runUuid: string;
  params: Record<string, KeyValueEntity>;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [filter, setFilter] = useState('');
  const autoExpandedRowsList = useRef<Record<string, boolean>>({});

  const paramsValues = useMemo(() => values(params), [params]);

  const paramsList = useMemo(
    () =>
      paramsValues.filter(({ key, value }) => {
        const filterLower = filter.toLowerCase();
        return key.toLowerCase().includes(filterLower) || value.toLowerCase().includes(filterLower);
      }),
    [filter, paramsValues],
  );

  const columns = useMemo<ParamsColumnDef[]>(
    () => [
      {
        id: 'key',
        accessorKey: 'key',
        header: () => (
          <FormattedMessage
            defaultMessage="Parameter"
            description="Run page > Overview > Parameters table > Key column header"
          />
        ),
        enableResizing: true,
        size: 240,
      },
      {
        id: 'value',
        header: () => (
          <FormattedMessage
            defaultMessage="Value"
            description="Run page > Overview > Parameters table > Value column header"
          />
        ),
        accessorKey: 'value',
        enableResizing: false,
        meta: { styles: { paddingLeft: 0 } },
        cell: ({ row: { original, getIsExpanded, toggleExpanded } }) => (
          <ExpandableParamValueCell
            name={original.key}
            value={original.value}
            isExpanded={getIsExpanded()}
            toggleExpanded={toggleExpanded}
            autoExpandedRowsList={autoExpandedRowsList.current}
          />
        ),
      },
    ],
    [],
  );

  const table = useReactTable({
    data: paramsList,
    getCoreRowModel: getCoreRowModel(),
    getExpandedRowModel: getExpandedRowModel(),
    getRowId: (row) => row.key,
    enableColumnResizing: true,
    columnResizeMode: 'onChange',
    columns,
  });

  const renderTableContent = () => {
    if (!paramsValues.length) {
      return (
        <div css={{ flex: '1', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Empty
            description={
              <FormattedMessage
                defaultMessage="No parameters recorded"
                description="Run page > Overview > Parameters table > No parameters recorded"
              />
            }
          />
        </div>
      );
    }

    const areAllResultsFiltered = paramsList.length < 1;

    return (
      <>
        <div css={{ marginBottom: theme.spacing.sm }}>
          <Input
            prefix={<SearchIcon />}
            placeholder={intl.formatMessage({
              defaultMessage: 'Search parameters',
              description: 'Run page > Overview > Parameters table > Filter input placeholder',
            })}
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            allowClear
          />
        </div>
        <Table
          scrollable
          empty={
            areAllResultsFiltered ? (
              <div css={{ marginTop: theme.spacing.md * 4 }}>
                <Empty
                  description={
                    <FormattedMessage
                      defaultMessage="No parameters match the search filter"
                      description="Run page > Overview > Parameters table > No results after filtering"
                    />
                  }
                />
              </div>
            ) : null
          }
        >
          <TableRow isHeader>
            {table.getLeafHeaders().map((header, index) => (
              <TableHeader
                key={header.id}
                resizable={header.column.getCanResize()}
                resizeHandler={header.getResizeHandler()}
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
            <TableRow key={row.id}>
              {row.getAllCells().map((cell) => (
                <TableCell
                  key={cell.id}
                  css={(cell.column.columnDef as ParamsColumnDef).meta?.styles}
                  style={{
                    flexGrow: cell.column.getCanResize() ? 0 : 1,
                    flexBasis: cell.column.getCanResize() ? cell.column.getSize() : undefined,
                  }}
                  multiline
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
    <div css={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <Typography.Title level={4}>
        <FormattedMessage
          defaultMessage="Parameters ({length})"
          description="Run page > Overview > Parameters table > Section title"
          values={{ length: paramsList.length }}
        />
      </Typography.Title>
      <div
        css={{
          padding: theme.spacing.sm,
          border: `1px solid ${theme.colors.borderDecorative}`,
          borderRadius: theme.general.borderRadiusBase,
          flex: 1,
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
