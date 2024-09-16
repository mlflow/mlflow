/**
 * Difference view plot implemented with AntD. Has additional requested features.
 */

import {
  DifferenceCardConfigCompareGroup,
  RunsChartsCardConfig,
  RunsChartsDifferenceCardConfig,
} from '../../runs-charts.types';
import type { RunsGroupByConfig } from '../../../experiment-page/utils/experimentPage.group-row-utils';
import { RunsChartsRunData } from '../RunsCharts.common';
import { FormattedMessage, useIntl } from 'react-intl';
import {
  ArrowDownIcon,
  ArrowUpIcon,
  Button,
  ChevronDownIcon,
  ChevronRightIcon,
  DashIcon,
  DropdownMenu,
  OverflowIcon,
  Tag,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { ReactNode, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  DifferenceChartCellDirection,
  differenceView,
  getDifferenceViewDataGroups,
  getDifferenceChartDisplayedValue,
  DIFFERENCE_CHART_DEFAULT_EMPTY_VALUE,
} from '../../utils/differenceView';
/**
 * We are using the antd Table component temporarily because it has accordions features built in while dubois does not.
 * We plan to deprecate and convert to dubois table when it is ready and supports nested accordions.
 */
import { Table } from 'antd';

import { RunColorPill } from '../../../experiment-page/components/RunColorPill';
import { Resizable, ResizableProps } from 'react-resizable';
import { useElementHeight } from '@mlflow/mlflow/src/common/utils/useElementHeight';

const HEADING_COLUMN_ID = 'headingColumn';
const DEFAULT_COLUMN_WIDTH = 200;

const ResizableTitle = (
  props: { width: number; onResize: ResizableProps['onResize'] } & React.ThHTMLAttributes<HTMLTableCellElement>,
) => {
  const { onResize, width, ...restProps } = props;
  if (!width) {
    return <th {...restProps} />;
  }

  return (
    <Resizable
      width={width}
      height={0}
      handle={
        <span
          css={{
            position: 'absolute',
            right: '-5px',
            bottom: '0',
            zIndex: '1',
            width: '10px',
            height: '100%',
            cursor: 'col-resize',
          }}
          onClick={(e) => {
            e.stopPropagation();
          }}
        />
      }
      onResize={onResize}
      draggableOpts={{ enableUserSelectHack: false }}
      css={{
        '.react-resizable': {
          position: 'relative',
          backgroundClip: 'padding-box',
        },
      }}
    >
      <th {...restProps} />
    </Resizable>
  );
};

interface RecordType {
  children: ReactNode;
}

const customExpandIcon = (props: {
  record: RecordType;
  expanded: boolean;
  onExpand: (record: RecordType, e: React.MouseEvent<HTMLElement>) => void;
}) => {
  if (!props.record.children) {
    // These custom padding is required to align the expand icon with the content
    return <div style={{ padding: '1px 14px', float: 'left' }} />;
  }
  return (
    <span>
      <Button
        icon={props.expanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
        onClick={(e) => {
          props.onExpand(props.record, e);
        }}
        componentId="mlflow.charts.difference_chart_expand_button"
        // The margin is required to align the expand icon in the button
        style={{ float: 'left', margin: '-2px 4px -2px 0px' }}
        size="small"
      />
    </span>
  );
};

/**
 * Transforms an array of objects into a format suitable for rendering in a table.
 * Each object in the array represents a row in the table.
 * If all values in a row are JSON objects with the same keys, the row is transformed into a parent row with child rows.
 * Each child row represents a key-value pair from the JSON objects.
 * If a value in a row is not a JSON object or the JSON objects don't have the same keys, the row is not transformed.
 *
 * @param data - An array of objects. Each object represents a row in the table.
 * @returns An array of objects. Each object represents a row or a parent row with child rows in the table.
 */
const getJSONRows = (data: { [key: string]: string | number }[]) => {
  const validateParseJSON = (value: string) => {
    try {
      const parsed = JSON.parse(value);
      if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed) || Object.keys(parsed).length === 0) {
        return null;
      }
      return parsed;
    } catch (e) {
      return null;
    }
  };

  const extractMaximumCommonSchema = (schema1: Record<any, any> | undefined, schema2: Record<any, any> | undefined) => {
    if (schema1 !== undefined && Object.keys(schema1).length === 0) {
      // This may not be a suitable object, return null
      return null;
    } else if (schema2 !== undefined && Object.keys(schema2).length === 0) {
      return null;
    }

    const schema: Record<string, unknown> = {};

    const collectKeys = (target: Record<any, any>, source: Record<any, any>) => {
      for (const key in source) {
        if (!target.hasOwnProperty(key) || target[key]) {
          if (typeof source[key] === 'object' && source[key] !== null && !Array.isArray(source[key])) {
            target[key] = target[key] || {};
            collectKeys(target[key], source[key]);
          } else if (source[key] === DIFFERENCE_CHART_DEFAULT_EMPTY_VALUE) {
            target[key] = true;
          } else {
            target[key] = false;
          }
        }
      }
    };

    schema1 !== undefined && collectKeys(schema, schema1);
    schema2 !== undefined && collectKeys(schema, schema2);

    return schema;
  };

  const getChildren = (
    parsedRowWithoutHeadingCol: { [key: string]: Record<any, any> | undefined },
    schema: Record<any, any>,
  ): Record<string, any>[] => {
    return Object.keys(schema).map((key) => {
      if (typeof schema[key] === 'boolean') {
        let result = {
          key: key,
          [HEADING_COLUMN_ID]: key,
        };
        Object.keys(parsedRowWithoutHeadingCol).forEach((runUuid) => {
          const value = parsedRowWithoutHeadingCol[runUuid]?.[key];
          result = {
            ...result,
            [runUuid]: value === undefined ? DIFFERENCE_CHART_DEFAULT_EMPTY_VALUE : value,
          };
        });
        return result;
      }
      // Recurse
      const newParsedRow: { [key: string]: Record<any, any> | undefined } = {};
      Object.keys(parsedRowWithoutHeadingCol).forEach((runUuid) => {
        newParsedRow[runUuid] = parsedRowWithoutHeadingCol[runUuid]?.[key];
      });

      return {
        key: key,
        [HEADING_COLUMN_ID]: key,
        children: getChildren(newParsedRow, schema[key]),
      };
    });
  };

  const isAllElementsJSON = (row: { [key: string]: string | number }) => {
    let jsonSchema: Record<any, any> | undefined = undefined;
    let isAllJson = true;
    const parsedRow: Record<string, any> = {};

    Object.keys(row).forEach((runUuid) => {
      if (runUuid !== HEADING_COLUMN_ID) {
        if (row[runUuid] !== DIFFERENCE_CHART_DEFAULT_EMPTY_VALUE) {
          const json = validateParseJSON(row[runUuid] as string);
          parsedRow[runUuid] = json;
          if (json === null) {
            isAllJson = false;
          } else {
            const commonSchema = extractMaximumCommonSchema(jsonSchema, json);
            if (commonSchema === null) {
              isAllJson = false;
            } else {
              jsonSchema = commonSchema;
            }
          }
        }
      }
    });
    if (isAllJson && jsonSchema !== undefined) {
      try {
        return {
          [HEADING_COLUMN_ID]: row[HEADING_COLUMN_ID],
          children: getChildren(parsedRow, jsonSchema),
          key: row[HEADING_COLUMN_ID],
        };
      } catch {
        return row;
      }
    } else {
      return row;
    }
  };
  return data.map(isAllElementsJSON);
};

const getTableBodyHeight = (tableHeight: number | undefined) => {
  if (!tableHeight) return 0;
  const headerHeight = 40;
  return tableHeight - headerHeight;
};

const antdTableComponents = {
  header: {
    cell: ResizableTitle,
  },
};

export const CellDifference = ({ label, direction }: { label: string; direction: DifferenceChartCellDirection }) => {
  const { theme } = useDesignSystemTheme();
  let paragraphColor = undefined;
  let icon = null;
  switch (direction) {
    case DifferenceChartCellDirection.NEGATIVE:
      paragraphColor = 'error';
      icon = <ArrowDownIcon color="danger" data-testid="negative-cell-direction" />;
      break;
    case DifferenceChartCellDirection.POSITIVE:
      paragraphColor = 'success';
      icon = <ArrowUpIcon color="success" data-testid="positive-cell-direction" />;
      break;
    case DifferenceChartCellDirection.SAME:
      paragraphColor = 'info';
      icon = <DashIcon css={{ color: theme.colors.textSecondary }} data-testid="same-cell-direction" />;
      break;
    default:
      break;
  }

  return (
    <div css={{ display: 'inline-flex', gap: theme.spacing.xs, alignItems: 'center' }}>
      <Typography.Paragraph color={paragraphColor} css={{ margin: 0 }}>
        {label}
      </Typography.Paragraph>
      {icon}
    </div>
  );
};

export const DifferenceViewPlotV2 = ({
  previewData,
  cardConfig,
  groupBy,
  setCardConfig,
}: {
  previewData: RunsChartsRunData[];
  cardConfig: RunsChartsDifferenceCardConfig;
  groupBy: RunsGroupByConfig | null;
  setCardConfig?: (setter: (current: RunsChartsCardConfig) => RunsChartsDifferenceCardConfig) => void;
}) => {
  const { formatMessage } = useIntl();
  const { theme } = useDesignSystemTheme();

  const { observeHeight, elementHeight } = useElementHeight();

  const { modelMetrics, systemMetrics, parameters, tags, attributes } = useMemo(
    () => getDifferenceViewDataGroups(previewData, cardConfig, HEADING_COLUMN_ID, groupBy),
    [previewData, cardConfig, groupBy],
  );

  // Add one length due to the heading column
  const [columnWidths, setColumnWidths] = useState(() => Array(previewData.length + 1).fill(DEFAULT_COLUMN_WIDTH));

  useEffect(() => {
    setColumnWidths(Array(previewData.length + 1).fill(DEFAULT_COLUMN_WIDTH));
  }, [previewData.length]);

  const { baselineColumn, nonBaselineColumns } = useMemo(() => {
    // baseline column (can be undefined if no baseline selected)
    let baselineColumn = previewData.find((runData) => runData.uuid === cardConfig.baselineColumnUuid);
    if (baselineColumn === undefined && previewData.length > 0) {
      // Set the first column as baseline column
      baselineColumn = previewData[0];
    }
    // non-baseline columns
    const nonBaselineColumns = previewData.filter(
      (runData) => baselineColumn === undefined || runData.uuid !== baselineColumn.uuid,
    );
    return { baselineColumn, nonBaselineColumns };
  }, [previewData, cardConfig.baselineColumnUuid]);

  const updateBaselineColumnUuid = useCallback(
    (baselineColumnUuid: string) => {
      if (setCardConfig) {
        setCardConfig((current) => ({
          ...(current as RunsChartsDifferenceCardConfig),
          baselineColumnUuid,
        }));
      }
    },
    [setCardConfig],
  );

  // Group each column by 5 groups

  const dataSource = cardConfig.compareGroups.reduce((acc: any[], group: DifferenceCardConfigCompareGroup) => {
    switch (group) {
      case DifferenceCardConfigCompareGroup.MODEL_METRICS:
        acc.push({
          [HEADING_COLUMN_ID]: formatMessage({
            defaultMessage: `Model Metrics`,
            description:
              'Experiment tracking > runs charts > cards > RunsChartsDifferenceChartCard > model metrics heading',
          }),
          children: [...modelMetrics],
          key: 'modelMetrics',
        });
        break;
      case DifferenceCardConfigCompareGroup.SYSTEM_METRICS:
        acc.push({
          [HEADING_COLUMN_ID]: formatMessage({
            defaultMessage: `System Metrics`,
            description:
              'Experiment tracking > runs charts > cards > RunsChartsDifferenceChartCard > system metrics heading',
          }),
          children: [...systemMetrics],
          key: 'systemMetrics',
        });
        break;
      case DifferenceCardConfigCompareGroup.PARAMETERS:
        acc.push({
          [HEADING_COLUMN_ID]: formatMessage({
            defaultMessage: `Parameters`,
            description:
              'Experiment tracking > runs charts > cards > RunsChartsDifferenceChartCard > parameters heading',
          }),
          children: getJSONRows(parameters),
          key: 'parameters',
        });
        break;
      case DifferenceCardConfigCompareGroup.ATTRIBUTES:
        acc.push({
          [HEADING_COLUMN_ID]: formatMessage({
            defaultMessage: `Attributes`,
            description:
              'Experiment tracking > runs charts > cards > RunsChartsDifferenceChartCard > attributes heading',
          }),
          children: [...attributes],
          key: 'attributes',
        });
        break;
      case DifferenceCardConfigCompareGroup.TAGS:
        acc.push({
          [HEADING_COLUMN_ID]: formatMessage({
            defaultMessage: `Tags`,
            description: 'Experiment tracking > runs charts > cards > RunsChartsDifferenceChartCard > tags heading',
          }),
          children: [...tags],
          key: 'tags',
        });
        break;
    }
    return acc;
  }, []);

  const columns = useMemo(() => {
    const handleResize =
      (index: number) =>
      (_: any, { size }: any) => {
        setColumnWidths((current) => {
          const next = [...current];
          next[index] = size.width;
          return next;
        });
      };

    const convertRunToColumnInfo = (runData: RunsChartsRunData, isBaseline: boolean, index: number) => {
      return {
        key: runData.uuid,
        dataIndex: runData.uuid,
        title: (
          <span css={{ display: 'flex', flexDirection: 'row', justifyContent: 'space-between' }}>
            <span css={{ display: 'inline-flex', gap: theme.spacing.sm, alignItems: 'center' }}>
              <RunColorPill color={runData.color} />
              {runData.displayName}
              {isBaseline && (
                <Tag
                  componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_charts_differenceviewplotv2.tsx_427"
                  css={{ margin: 0 }}
                >
                  <FormattedMessage
                    defaultMessage="baseline"
                    description="Runs charts > components > charts > DifferenceViewPlot > baseline tag"
                  />
                </Tag>
              )}
            </span>
          </span>
        ),
        render: (value: string | number, record: any) => {
          if (isBaseline) {
            return getDifferenceChartDisplayedValue(value);
          }
          if (value === undefined) {
            return null;
          }
          const rowDifference = baselineColumn ? differenceView(value, record[baselineColumn.uuid]) : null;
          return (
            <span css={{ display: 'inline-flex', gap: theme.spacing.md, verticalAlign: 'middle' }}>
              <Typography.Text>{getDifferenceChartDisplayedValue(value)}</Typography.Text>
              {baselineColumn && cardConfig.showChangeFromBaseline && rowDifference && (
                <CellDifference label={rowDifference.label} direction={rowDifference.direction} />
              )}
            </span>
          );
        },
        ellipsis: true,
        width: columnWidths[index],
        onHeaderCell: (column: any) => ({
          width: column.width,
          onResize: handleResize(index),
          style: { background: 'transparent' },
        }),
        filterIcon: (
          <DropdownMenu.Root>
            <DropdownMenu.Trigger asChild>
              <OverflowIcon />
            </DropdownMenu.Trigger>
            <DropdownMenu.Content>
              <DropdownMenu.Item
                componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_charts_differenceviewplotv2.tsx_467"
                onClick={() => updateBaselineColumnUuid(runData.uuid)}
              >
                <FormattedMessage
                  defaultMessage="Set as baseline"
                  description="Runs charts > components > charts > DifferenceViewPlot > Set as baseline dropdown option"
                />
              </DropdownMenu.Item>
            </DropdownMenu.Content>
          </DropdownMenu.Root>
        ),
        // This is needed to let the filter iconshow up.
        filterDropdown: () => null,
      };
    };

    return [
      {
        dataIndex: HEADING_COLUMN_ID,
        title: formatMessage({
          defaultMessage: 'Compare by',
          description: 'Runs charts > components > charts > DifferenceViewPlot > Compare by column heading',
        }),
        key: HEADING_COLUMN_ID,
        ellipsis: true,
        fixed: true,
        width: columnWidths[0],
        onHeaderCell: (column: any) => ({
          width: column.width,
          onResize: handleResize(0),
          style: { background: 'white' }, // White is needed to block other columns when scrolling
        }),
      },
      ...(baselineColumn ? [convertRunToColumnInfo(baselineColumn, true, 1)] : []),
      ...nonBaselineColumns.map((runData, index) => convertRunToColumnInfo(runData, false, index + 2)),
    ];
  }, [
    theme.spacing,
    formatMessage,
    baselineColumn,
    nonBaselineColumns,
    cardConfig.showChangeFromBaseline,
    columnWidths,
    setColumnWidths,
    updateBaselineColumnUuid,
  ]);

  return (
    <div
      css={{
        display: 'flex',
        overflow: 'auto hidden',
        cursor: 'pointer',
        height: '100%',
        width: '100%',
      }}
      ref={observeHeight}
    >
      <Table
        components={antdTableComponents}
        size="small"
        columns={columns}
        dataSource={dataSource}
        scroll={{ y: getTableBodyHeight(elementHeight) }}
        pagination={false}
        expandable={{
          expandIcon: customExpandIcon,
          defaultExpandAllRows: true,
          indentSize: 15,
        }}
        sticky
      />
    </div>
  );
};
