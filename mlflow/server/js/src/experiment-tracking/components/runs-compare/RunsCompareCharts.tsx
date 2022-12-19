import { Button, OverflowIcon, Typography, Spacer, DropdownMenu } from '@databricks/design-system';
import { Theme } from '@emotion/react';
import { CompareRunsChartSetup } from '../../types';
import { RunRowType } from '../experiment-page/utils/experimentPage.row-types';

export interface RunsCompareChartsProps {
  comparedRuns: RunRowType[];
  chartsConfig: CompareRunsChartSetup[];
  onRemoveChart: (chart: CompareRunsChartSetup) => void;
}

// prettier-ignore
// eslint-disable-next-line prettier/prettier
/**
 * Sample SVG component for visualization purposes
 */
const ChartSamplePlaceholderData = () => <svg width='auto' viewBox='0 0 479 235' fill='none' xmlns='http://www.w3.org/2000/svg'>    <path d='M9.41016 233.928L60.5718 214.28L110.299 187.443L164.329 172.586L185.846 144.312L221.707 110.765L248.005 95.43L290.082 73.8645' stroke='#FF3621' strokeWidth='2' strokeLinecap='round' strokeLinejoin='round' />    <path d='M9.41016 233.928L42.3472 231.041L74.3607 227.097L105.713 223.545L122.997 220.758L191.095 214.264L301.297 209.945L454.19 206.821' stroke='#077A9D' strokeWidth='2' strokeLinecap='round' strokeLinejoin='round' />    <path d='M9.41016 233.928L25.5143 206.821L48.5201 177.777L62.3236 150.024L92.3814 123.562L142.827 88.9139L164.909 52.5656L228.733 22.231' stroke='#FCA4A1' strokeWidth='2' strokeLinecap='round' strokeLinejoin='round' />    <line x1='6.84375' y1='0.286865' x2='6.84376' y2='232.637' stroke='#BDCDDB' />    <line x1='0.208984' y1='233.428' x2='7.87761' y2='233.428' stroke='#BDCDDB' />    <line x1='0.208984' y1='207.612' x2='7.87761' y2='207.612' stroke='#BDCDDB' />    <line x1='0.208984' y1='181.795' x2='7.87761' y2='181.795' stroke='#BDCDDB' />    <line x1='0.208984' y1='155.978' x2='7.87761' y2='155.978' stroke='#BDCDDB' />    <line x1='0.208984' y1='130.161' x2='7.87761' y2='130.161' stroke='#BDCDDB' />    <line x1='0.208984' y1='104.345' x2='7.87761' y2='104.345' stroke='#BDCDDB' />    <line x1='0.208984' y1='78.5278' x2='7.87761' y2='78.5278' stroke='#BDCDDB' />    <line x1='0.208984' y1='52.7112' x2='7.87761' y2='52.7112' stroke='#BDCDDB' />    <line x1='0.208984' y1='26.8945' x2='7.87761' y2='26.8945' stroke='#BDCDDB' />    <line x1='0.208984' y1='1.07764' x2='7.87761' y2='1.07764' stroke='#BDCDDB' />    <line x1='478.73' y1='233.137' x2='6.34338' y2='233.137' stroke='#BDCDDB' />  </svg>;

/**
 * Sample chart component for visualization purposes
 * TODO: implement actual cards for various chart types
 */
const SampleChartCard = ({
  config,
  onDelete,
}: {
  onDelete: () => void;
  config: CompareRunsChartSetup;
}) => (
  <div css={styles.chartEntry}>
    <div css={styles.chartEntryTitle}>
      <Typography.Title level={4}>
        Chart {config.type} for {config.metricKey}
      </Typography.Title>
      <DropdownMenu.Root>
        <DropdownMenu.Trigger asChild>
          <Button type='tertiary' icon={<OverflowIcon />} />
        </DropdownMenu.Trigger>
        <DropdownMenu.Content align='end' minWidth={100}>
          <DropdownMenu.Item
            onClick={() => {
              /* TODO: reconfiguring existing charts */
            }}
          >
            Configure
          </DropdownMenu.Item>
          <DropdownMenu.Item onClick={onDelete}>Delete</DropdownMenu.Item>
        </DropdownMenu.Content>
      </DropdownMenu.Root>
    </div>
    <div css={(theme) => ({ fontSize: 10, color: theme.colors.grey600 })}>UUID: {config.uuid}</div>
    <Spacer size='md' />
    <ChartSamplePlaceholderData />
  </div>
);

export const RunsCompareCharts = ({ chartsConfig, onRemoveChart }: RunsCompareChartsProps) => {
  return (
    <div css={styles.chartsWrapper}>
      {chartsConfig.map((chartConfig) => (
        <SampleChartCard
          key={chartConfig.uuid}
          config={chartConfig}
          onDelete={() => onRemoveChart(chartConfig)}
        />
      ))}
    </div>
  );
};

const styles = {
  chartsWrapper: (theme: Theme) => ({
    display: 'grid' as const,
    gridTemplateColumns: 'repeat(auto-fill, minmax(400px, 1fr))',
    gap: theme.spacing.md,
  }),
  chartEntryTitle: (theme: Theme) => ({
    display: 'grid' as const,
    gridTemplateColumns: '1fr auto auto',
    height: theme.general.heightSm,
    alignItems: 'center',
  }),
  chartEntry: (theme: Theme) => ({
    backgroundColor: theme.colors.backgroundPrimary,
    padding: theme.spacing.md,
    border: `1px solid ${theme.colors.border}`,
    borderRadius: theme.general.borderRadiusBase,
  }),
};
