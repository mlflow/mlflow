import { useDesignSystemTheme, TableRow, TableHeader, TableCell, Table } from '@databricks/design-system';
import { useMemo, useRef, useState, useEffect } from 'react';
import type { RunsChartsImageCardConfig, RunsChartsCardConfig } from '../../runs-charts.types';
import type { RunsChartsRunData } from '../RunsCharts.common';
import { EmptyImageGridPlot, ImagePlotWithHistory, ImageGridRunHeader } from './ImageGridPlot.common';
import type { ImageEntity } from '@mlflow/mlflow/src/experiment-tracking/types';
import { FormattedMessage } from 'react-intl';

const LABEL_COLUMN_WIDTH = 160;
const MIN_CELL_SIZE = 150;

export const ImageGridMultipleKeyPlot = ({
  previewData,
  cardConfig,
}: {
  previewData: RunsChartsRunData[];
  cardConfig: RunsChartsImageCardConfig;
  groupBy?: string;
  setCardConfig?: (setter: (current: RunsChartsCardConfig) => RunsChartsImageCardConfig) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  // We can't use useDynamicPlotSize or useResizeObserver directly on the table
  // container because cellSize changes feed back into the table layout, creating
  // a resize loop. Instead we observe a zero-height measurement div and use a
  // dead zone to ignore small fluctuations.
  const measureRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState(0);
  const lastWidth = useRef(0);

  useEffect(() => {
    if (!measureRef.current || !window.ResizeObserver) return;
    let rafId = 0;
    const observer = new ResizeObserver(([entry]) => {
      cancelAnimationFrame(rafId);
      rafId = requestAnimationFrame(() => {
        const newWidth = Math.floor(entry.contentRect.width);
        if (newWidth > 0 && Math.abs(newWidth - lastWidth.current) > 10) {
          lastWidth.current = newWidth;
          setContainerWidth(newWidth);
        }
      });
    });
    observer.observe(measureRef.current);
    return () => {
      cancelAnimationFrame(rafId);
      observer.disconnect();
    };
  }, []);

  const displayRuns = previewData.filter((run: RunsChartsRunData) => Object.keys(run.images).length !== 0);

  const cellSize = useMemo(() => {
    if (!containerWidth || displayRuns.length === 0) return MIN_CELL_SIZE;
    const availableWidth = containerWidth - LABEL_COLUMN_WIDTH - theme.spacing.md * (displayRuns.length + 1);
    return Math.max(MIN_CELL_SIZE, Math.floor(availableWidth / displayRuns.length));
  }, [containerWidth, displayRuns.length, theme.spacing.md]);

  if (displayRuns.length === 0) {
    return <EmptyImageGridPlot />;
  }
  return (
    <div css={{ height: '100%', width: '100%' }}>
      <div ref={measureRef} css={{ height: 0, overflow: 'hidden' }} />
      <Table grid scrollable>
        <TableRow isHeader>
          <TableHeader
            componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_charts_imagegridmultiplekeyplot.tsx_44"
            css={{ minWidth: LABEL_COLUMN_WIDTH }}
          >
            <FormattedMessage
              defaultMessage="images"
              description="Experiment tracking > runs charts > charts > image grid multiple key > table header text"
            />
          </TableHeader>
          {displayRuns.map((run: RunsChartsRunData) => {
            return (
              <TableHeader
                componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_charts_imagegridmultiplekeyplot.tsx_52"
                key={run.uuid}
                css={{ minWidth: cellSize, fontWeight: 'normal' }}
              >
                <ImageGridRunHeader
                  displayName={run.displayName}
                  color={run.color}
                  params={run.params}
                  maxParamsWidth={cellSize}
                />
              </TableHeader>
            );
          })}
        </TableRow>
        {cardConfig.imageKeys.map((imageKey) => {
          return (
            <TableRow key={imageKey}>
              <TableCell css={{ minWidth: LABEL_COLUMN_WIDTH }}>
                <div style={{ whiteSpace: 'normal' }}>{imageKey}</div>
              </TableCell>
              {displayRuns.map((run: RunsChartsRunData) => {
                if (run.images[imageKey] && Object.keys(run.images[imageKey]).length > 0) {
                  const metadataByStep = Object.values(run.images[imageKey]).reduce(
                    (acc, metadata) => {
                      if (metadata.step !== undefined) {
                        acc[metadata.step] = metadata;
                      }
                      return acc;
                    },
                    {} as Record<number, ImageEntity>,
                  );
                  return (
                    <TableCell
                      key={run.uuid}
                      css={{
                        minWidth: cellSize,
                        '&:hover': {
                          backgroundColor: theme.colors.tableBackgroundUnselectedHover,
                        },
                      }}
                    >
                      <ImagePlotWithHistory metadataByStep={metadataByStep} step={cardConfig.step} runUuid={run.uuid} />
                    </TableCell>
                  );
                }
                return <TableCell key={run.uuid} css={{ minWidth: cellSize }} />;
              })}
            </TableRow>
          );
        })}
      </Table>
    </div>
  );
};
