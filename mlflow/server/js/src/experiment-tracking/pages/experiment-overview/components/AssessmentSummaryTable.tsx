import React, { useMemo } from 'react';
import { Typography, useDesignSystemTheme, SparkleIcon } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { OverviewChartHeader, OverviewChartContainer } from './OverviewChartComponents';
import { formatCount, useChartColors } from '../utils/chartUtils';
import { useSortState, useSummaryTableStyles, SortableHeader, NameCellWithColor } from './SummaryTableComponents';

type SortColumn = 'scorerName' | 'totalCount' | 'avgValue';

interface AssessmentSummaryTableProps {
  /** List of assessment/scorer names */
  assessmentNames: string[];
  /** Map of assessment name to its total count */
  countsByName: Map<string, number>;
  /** Map of assessment name to its average value (only for numeric assessments) */
  avgValuesByName: Map<string, number>;
}

/**
 * Assessment Summary Table component displaying per-scorer metrics in a table-like view.
 * Shows scorer name, total count, and average value for each assessment.
 * Non-numeric assessments show '-' for average value.
 */
export const AssessmentSummaryTable: React.FC<AssessmentSummaryTableProps> = ({
  assessmentNames,
  countsByName,
  avgValuesByName,
}) => {
  const { theme } = useDesignSystemTheme();
  const { getChartColor } = useChartColors();
  const { sortColumn, sortDirection, handleSort } = useSortState<SortColumn>('totalCount');
  const { headerRowStyle, bodyRowStyle, cellStyle } = useSummaryTableStyles('minmax(80px, 2fr) 1fr 1fr');

  // Transform data into sortable rows
  const tableData = useMemo(() => {
    return assessmentNames.map((name) => ({
      scorerName: name,
      totalCount: countsByName.get(name)!,
      avgValue: avgValuesByName.get(name),
    }));
  }, [assessmentNames, countsByName, avgValuesByName]);

  // Sort the data
  const sortedTableData = useMemo(() => {
    if (!tableData.length) return tableData;

    return [...tableData].sort((a, b) => {
      let comparison = 0;
      switch (sortColumn) {
        case 'scorerName':
          comparison = a.scorerName.localeCompare(b.scorerName);
          break;
        case 'totalCount':
          comparison = a.totalCount - b.totalCount;
          break;
        case 'avgValue': {
          // Handle undefined values - treat them as lower than any number
          const aVal = a.avgValue ?? -Infinity;
          const bVal = b.avgValue ?? -Infinity;
          comparison = aVal - bVal;
          break;
        }
      }
      return sortDirection === 'asc' ? comparison : -comparison;
    });
  }, [tableData, sortColumn, sortDirection]);

  // Format average value: show number with 2 decimal places or '-' for non-numeric
  const formatAvgValue = (value: number | undefined): string => {
    if (value === undefined) return '-';
    return value.toFixed(2);
  };

  return (
    <OverviewChartContainer componentId="mlflow.overview.quality.quality_summary_table">
      <OverviewChartHeader
        icon={<SparkleIcon css={{ color: theme.colors.yellow500 }} />}
        title={
          <FormattedMessage
            defaultMessage="Quality Summary"
            description="Title for the quality summary table section"
          />
        }
      />

      <div css={{ display: 'flex', flexDirection: 'column' }}>
        {/* Table header */}
        <div css={headerRowStyle}>
          <SortableHeader column="scorerName" sortColumn={sortColumn} sortDirection={sortDirection} onSort={handleSort}>
            <FormattedMessage defaultMessage="Scorer" description="Column header for scorer name" />
          </SortableHeader>
          <SortableHeader
            column="totalCount"
            sortColumn={sortColumn}
            sortDirection={sortDirection}
            onSort={handleSort}
            centered
          >
            <FormattedMessage defaultMessage="Total Count" description="Column header for total count" />
          </SortableHeader>
          <SortableHeader
            column="avgValue"
            sortColumn={sortColumn}
            sortDirection={sortDirection}
            onSort={handleSort}
            centered
          >
            <FormattedMessage defaultMessage="Average Value" description="Column header for average value" />
          </SortableHeader>
        </div>

        {/* Scrollable table body */}
        <div css={{ maxHeight: 300, overflowY: 'auto' }}>
          {sortedTableData.map((row, index) => (
            <div key={row.scorerName} css={bodyRowStyle}>
              <NameCellWithColor name={row.scorerName} color={getChartColor(index)} />
              <Typography.Text css={cellStyle}>{formatCount(row.totalCount)}</Typography.Text>
              <Typography.Text css={cellStyle}>{formatAvgValue(row.avgValue)}</Typography.Text>
            </div>
          ))}
        </div>
      </div>
    </OverviewChartContainer>
  );
};
