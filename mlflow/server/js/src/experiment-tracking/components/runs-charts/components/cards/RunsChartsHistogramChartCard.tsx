import { useState, useEffect, useMemo } from 'react';
import { getAjaxUrl } from '../../../../../common/utils/FetchUtils';
import type { RunsChartsRunData } from '../RunsCharts.common';
import type { RunsChartsHistogramCardConfig } from '../../runs-charts.types';
import { RunsHistogram3DPlot, type HistogramData } from '../RunsHistogram3DPlot';
import type { RunsChartCardFullScreenProps, RunsChartCardVisibilityProps } from './ChartCard.common';
import {
  RunsChartCardWrapper,
  type RunsChartCardReorderProps,
  RunsChartsChartsDragGroup,
  RunsChartCardLoadingPlaceholder,
} from './ChartCard.common';
import { RunsChartsNoDataFoundIndicator } from '../RunsChartsNoDataFoundIndicator';

export interface RunsChartsHistogramChartCardProps
  extends RunsChartCardReorderProps,
    RunsChartCardFullScreenProps,
    RunsChartCardVisibilityProps {
  config: RunsChartsHistogramCardConfig;
  chartRunData: RunsChartsRunData[];
  hideEmptyCharts?: boolean;
  onDelete: () => void;
  onEdit: () => void;
}

/**
 * Fetches histogram data from artifacts for a given run and histogram key.
 * Each step is stored as a separate JSON file under histograms/{key}/step_{n}.json.
 */
const fetchHistogramData = async (runId: string, histogramKey: string): Promise<HistogramData[]> => {
  const sanitizedKey = histogramKey.replace(/\//g, '_');
  const artifactDir = `histograms/${sanitizedKey}`;

  // List all step files in the histogram directory
  const listResponse = await fetch(
    getAjaxUrl(`ajax-api/2.0/mlflow/artifacts/list?run_id=${runId}&path=${encodeURIComponent(artifactDir)}`),
  );

  if (!listResponse.ok) {
    return [];
  }

  const listData = await listResponse.json();
  const files = (listData.files || []).filter((f: any) => f.path?.endsWith('.json') && !f.is_dir);

  // Fetch each step file in parallel
  const histograms = await Promise.all(
    files.map(async (f: any) => {
      const response = await fetch(getAjaxUrl(`get-artifact?path=${encodeURIComponent(f.path)}&run_uuid=${runId}`));
      if (!response.ok) {
        return null;
      }
      return response.json();
    }),
  );

  return histograms.filter((h): h is HistogramData => h !== null);
};

export const RunsChartsHistogramChartCard = ({
  config,
  chartRunData,
  onDelete,
  onEdit,
  fullScreen,
  setFullScreenChart,
  hideEmptyCharts,
  isInViewport: isInViewportProp,
  ...reorderProps
}: RunsChartsHistogramChartCardProps) => {
  const [histograms, setHistograms] = useState<HistogramData[]>([]);
  const [loading, setLoading] = useState(true);

  const toggleFullScreenChart = () => {
    setFullScreenChart?.({
      config,
      title: config.displayName || `Histogram: ${config.histogramKeys[0]}`,
      subtitle: null,
    });
  };

  const isInViewport = fullScreen || isInViewportProp;

  const selectedRuns = useMemo(() => {
    if (config.selectedRunUuids && config.selectedRunUuids.length > 0) {
      return chartRunData.filter((run) => config.selectedRunUuids.includes(run.uuid));
    }
    return chartRunData.slice(0, 1);
  }, [chartRunData, config.selectedRunUuids]);

  useEffect(() => {
    const fetchData = async () => {
      if (config.histogramKeys.length === 0 || selectedRuns.length === 0) {
        setLoading(false);
        return;
      }

      setLoading(true);

      try {
        const firstKey = config.histogramKeys[0];
        const firstRun = selectedRuns[0];
        const data = await fetchHistogramData(firstRun.uuid, firstKey);
        setHistograms(data);
      } catch {
        setHistograms([]);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [config.histogramKeys, selectedRuns]);

  const title = useMemo(() => {
    if (config.displayName) {
      return config.displayName;
    }
    if (config.histogramKeys.length > 0) {
      return `Histogram: ${config.histogramKeys[0]}`;
    }
    return 'Histogram Distribution';
  }, [config.displayName, config.histogramKeys]);

  const isEmptyDataset = config.histogramKeys.length === 0 || histograms.length === 0;

  const chartBody = (
    <div css={styles.histogramChartWrapper}>
      {!isInViewport ? null : loading ? (
        <RunsChartCardLoadingPlaceholder />
      ) : (
        <RunsHistogram3DPlot histograms={histograms} logScale={false} />
      )}
    </div>
  );

  if (fullScreen) {
    return chartBody;
  }

  if (hideEmptyCharts && isEmptyDataset) {
    return null;
  }

  return (
    <RunsChartCardWrapper
      onEdit={onEdit}
      onDelete={onDelete}
      title={title}
      uuid={config.uuid}
      dragGroupKey={RunsChartsChartsDragGroup.GENERAL_AREA}
      toggleFullScreenChart={isEmptyDataset ? undefined : toggleFullScreenChart}
      isHidden={!isInViewport}
      {...reorderProps}
    >
      {isEmptyDataset ? <RunsChartsNoDataFoundIndicator /> : chartBody}
    </RunsChartCardWrapper>
  );
};

const styles = {
  histogramChartWrapper: {
    overflow: 'hidden',
    height: '100%',
  },
};

export default RunsChartsHistogramChartCard;
