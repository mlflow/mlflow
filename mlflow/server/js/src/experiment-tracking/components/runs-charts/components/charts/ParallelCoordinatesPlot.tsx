import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useDesignSystemTheme } from '@databricks/design-system';
import Parcoords from 'parcoord-es';
import 'parcoord-es/dist/parcoords.css';
import { scaleSequential } from 'd3-scale';
import { useDynamicPlotSize } from '../RunsCharts.common';
import './ParallelCoordinatesPlot.css';
import { truncateChartMetricString } from '../../../../utils/MetricsUtils';
import { useRunsChartTraceHighlight } from '../../hooks/useRunsChartTraceHighlight';
import { RunsChartCardLoadingPlaceholder } from '../cards/ChartCard.common';

/**
 * Attaches custom tooltip to the axis label inside SVG
 */
const attachCustomTooltip = (toolTipClass: string, labelText: string, targetLabel: Element) => {
  const tooltipPadding = 4;
  const svgNS = 'http://www.w3.org/2000/svg';
  const tooltipGroup = document.createElementNS(svgNS, 'g');
  const newRect = document.createElementNS(svgNS, 'rect');
  const newText = document.createElementNS(svgNS, 'text');
  newText.innerHTML = labelText;
  newText.setAttribute('fill', 'black');
  tooltipGroup.classList.add(toolTipClass);
  tooltipGroup.appendChild(newRect);
  tooltipGroup.appendChild(newText);
  targetLabel.parentNode?.insertBefore(tooltipGroup, targetLabel.nextSibling);

  const textBBox = newText.getBBox();

  newRect.setAttribute('x', (textBBox.x - tooltipPadding).toString());
  newRect.setAttribute('y', (textBBox.y - tooltipPadding).toString());
  newRect.setAttribute('width', (textBBox.width + 2 * tooltipPadding).toString());
  newRect.setAttribute('height', (textBBox.height + 2 * tooltipPadding).toString());
  newRect.setAttribute('fill', 'white');
};

const ParallelCoordinatesPlotImpl = (props: {
  data: any;
  metricKey: string;
  selectedParams: string[];
  selectedMetrics: string[];
  onHover: (runUuid?: string) => void;
  onUnhover: () => void;
  closeContextMenu: () => void;
  width: number;
  height: number;
  axesRotateThreshold: number;
  selectedRunUuid: string | null;
}) => {
  // De-structure props here so they will be easily used
  // as hook dependencies later on
  const {
    onHover,
    onUnhover,
    selectedRunUuid,
    data,
    axesRotateThreshold,
    selectedMetrics,
    selectedParams,
    width: chartWidth,
    height: chartHeight,
    closeContextMenu,
  } = props;
  const chartRef = useRef<HTMLDivElement>(null);
  const parcoord: any = useRef<null>();

  // Keep the state of the actually hovered run internally
  const [hoveredRunUuid, setHoveredRunUuid] = useState('');

  // Basing on the stateful hovered run uuid, call tooltip-related callbacks
  useEffect(() => {
    if (hoveredRunUuid) {
      onHover?.(hoveredRunUuid);
    } else {
      onUnhover?.();
    }
  }, [hoveredRunUuid, onHover, onUnhover]);

  // Memoize this function so it won't cause dependency re-triggers
  const getActiveData = useCallback(() => {
    if (parcoord.current.brushed() !== false) return parcoord.current.brushed();
    return parcoord.current.data();
  }, []);

  const { onHighlightChange } = useRunsChartTraceHighlight();

  // Listener that will be called when the highlight changes
  const highlightListener = useCallback(
    (traceUuid: string | null) => {
      if (!traceUuid) {
        parcoord.current.unhighlight();
        return;
      }
      // Get immediate displayed runs data
      const displayedData: { uuid: string; [k: string]: number | string }[] = getActiveData();

      const runsToHighlight = displayedData.filter(({ uuid }) => traceUuid === uuid);

      if (runsToHighlight.length) {
        parcoord.current.highlight(runsToHighlight);
      } else {
        parcoord.current.unhighlight();
      }
    },
    [getActiveData],
  );

  useEffect(() => onHighlightChange(highlightListener), [onHighlightChange, highlightListener]);

  // Basing on the stateful hovered run uuid and selected run uuid, determine
  // which runs should be highlighted
  useEffect(() => {
    if (!parcoord.current) {
      return;
    }
    // Get immediate active data
    const activeData = getActiveData();

    // Get all (at most two) runs that are highlighted and/or selected
    const runsToHighlight = activeData.filter((d: any) => [hoveredRunUuid, selectedRunUuid].includes(d.uuid));

    // Either select them or unselect all
    if (runsToHighlight.length) {
      parcoord.current.highlight(runsToHighlight);
    } else {
      parcoord.current.unhighlight();
    }
  }, [hoveredRunUuid, selectedRunUuid, getActiveData]);

  const getClickedLines = useCallback(
    (mouseLocation: { offsetX: number; offsetY: number }) => {
      const clicked: [number, number][] = [];
      const activeData = getActiveData();
      if (activeData.length === 0) return false;

      const graphCentPts = getCentroids();
      if (graphCentPts.length === 0) return false;

      // find between which axes the point is
      const potentialAxeNum: number | boolean = findAxes(mouseLocation, graphCentPts[0]);
      if (!potentialAxeNum) return false;
      const axeNum: number = potentialAxeNum;

      graphCentPts.forEach((d: [number, number][], i: string | number) => {
        if (isOnLine(d[axeNum - 1], d[axeNum], mouseLocation, 2)) {
          clicked.push(activeData[i]);
        }
      });

      return [clicked];
    },
    [getActiveData],
  );

  const highlightLineOnHover = useCallback(
    (mouseLocation: { offsetX: number; offsetY: number }) => {
      // compute axes locations
      const axes_left_bounds: number[] = [];
      let axes_width = 0;
      const wrapperElement = chartRef.current;

      if (!wrapperElement) {
        return;
      }

      wrapperElement.querySelectorAll('.dimension').forEach(function getAxesBounds(element) {
        const transformValue = element.getAttribute('transform');
        // transformValue is a string like "transform(100)"
        if (transformValue) {
          const parsedTransformValue = parseFloat(transformValue.split('(')[1].split(')')[0]);
          axes_left_bounds.push(parsedTransformValue);
          const { width } = (element as SVGGraphicsElement).getBBox();
          if (axes_width === 0) axes_width = width;
        }
      });

      const axes_locations: any[] = [];
      for (let i = 0; i < axes_left_bounds.length; i++) {
        axes_locations.push(axes_left_bounds[i] + axes_width / 2);
      }

      let clicked = [];

      const clickedData = getClickedLines(mouseLocation);

      let foundRunUuid = '';
      if (clickedData && clickedData[0].length !== 0) {
        [clicked] = clickedData;

        if (clicked.length > 1) {
          clicked = [clicked[1]];
        }

        // check if the mouse is over an axis with tolerance of 10px
        if (axes_locations.some((x) => Math.abs(x - mouseLocation.offsetX) < 10)) {
          // We are hovering over axes, do nothing
        } else {
          const runData: any = clicked[0];
          foundRunUuid = runData['uuid'];
        }
      }
      setHoveredRunUuid(foundRunUuid);
    },
    [chartRef, getClickedLines],
  );

  const getCentroids = () => {
    const margins = parcoord.current.margin();
    const brushedData = parcoord.current.brushed().length ? parcoord.current.brushed() : parcoord.current.data();

    return brushedData.map((d: any) => {
      const centroidPoints = parcoord.current.compute_real_centroids(d);
      return centroidPoints.map((p: any[]) => [p[0] + margins.left, p[1] + margins.top]);
    });
  };

  const findAxes = (testPt: { offsetX: number; offsetY: number }, cenPts: string | any[]) => {
    // finds between which two axis the mouse is
    const x = testPt.offsetX;

    // make sure it is inside the range of x
    if (cenPts[0][0] > x) return false;
    if (cenPts[cenPts.length - 1][0] < x) return false;

    // find between which segment the point is
    for (let i = 0; i < cenPts.length; i++) {
      if (cenPts[i][0] > x) return i;
    }
    return false;
  };

  const isOnLine = (
    startPt: [number, number],
    endPt: [number, number],
    testPt: { offsetX: number; offsetY: number },
    tol: number,
  ) => {
    // check if test point is close enough to a line
    // between startPt and endPt. close enough means smaller than tolerance
    const x0 = testPt.offsetX;
    const y0 = testPt.offsetY;
    const [x1, y1] = startPt;
    const [x2, y2] = endPt;
    const Dx = x2 - x1;
    const Dy = y2 - y1;
    const delta = Math.abs(Dy * x0 - Dx * y0 - x1 * y2 + x2 * y1) / Math.sqrt(Math.pow(Dx, 2) + Math.pow(Dy, 2));
    if (delta <= tol) return true;
    return false;
  };

  useEffect(() => {
    if (chartRef !== null) {
      const num_axes = selectedParams.length + selectedMetrics.length;
      const axesLabelTruncationThreshold = num_axes > axesRotateThreshold ? 15 : 15;
      const tickLabelTruncationThreshold = num_axes > axesRotateThreshold ? 9 : 9;
      const maxAxesLabelWidth = (chartWidth / num_axes) * 0.8;
      const maxTickLabelWidth = (chartWidth / num_axes) * 0.4;

      // last element of selectedMetrics is the primary metric
      const metricKey: string = selectedMetrics[selectedMetrics.length - 1];
      // iterate through runs in data to find max and min of metricKey
      const metricVals = data.map((run: any) => run[metricKey]);
      const minValue = Math.min(...metricVals.filter((v: number) => !isNaN(v)));
      const maxValue = Math.max(...metricVals.filter((v: number) => !isNaN(v)));

      // use d3 scale to map metric values to colors
      // color math is from interpolateTurbo in d3-scale-chromatic https://github.com/d3/d3-scale-chromatic/blob/main/src/sequential-multi/turbo.js
      // prettier-ignore
      const color_set = scaleSequential()
        .domain([minValue, maxValue])
        .interpolator((x) => {
          const t = Math.max(0, Math.min(1, x));
          return `rgb(
            ${Math.max(0, Math.min(255, Math.round(34.61 + t * (1172.33 - t * (10793.56 - t * (33300.12 - t * (38394.49 - t * 14825.05)))))))},
            ${Math.max(0, Math.min(255, Math.round(23.31 + t * (557.33 + t * (1225.33 - t * (3574.96 - t * (1073.77 + t * 707.56)))))))},
            ${Math.max(0, Math.min(255, Math.round(27.2 + t * (3211.1 - t * (15327.97 - t * (27814 - t * (22569.18 - t * 6838.66)))))))}
          )`;
        });

      const wrapperElement = chartRef.current;

      // clear the existing chart state
      if (wrapperElement) {
        wrapperElement.querySelector('#wrapper svg')?.remove();
      }

      // clear old canvases if they exist
      if (wrapperElement) {
        wrapperElement.querySelectorAll('canvas').forEach((canvas) => canvas.remove());
      }
      const getAxesTypes = () => {
        const keys = Object.keys(data[0]);
        const nonNullValues = keys.map((key) => data.map((d: any) => d[key]).filter((v: any) => v !== null));
        const types = nonNullValues.map((v: any) => {
          if (v.every((x: any) => !isNaN(x) && x !== null)) return 'number';
          return 'string';
        });
        return Object.fromEntries(keys.map((_, i) => [keys[i], { type: types[i] }]));
      };

      parcoord.current = Parcoords()(chartRef.current)
        .width(chartWidth)
        .height(chartHeight)
        .data(data)
        .dimensions(getAxesTypes())
        .alpha(0.8)
        .alphaOnBrushed(0.1)
        .hideAxis(['uuid'])
        .lineWidth(1)
        .color((d: any) => {
          if (d && metricKey in d && d[metricKey] !== 'null') {
            return color_set(d[metricKey]);
          } else {
            return '#f33';
          }
        })
        .createAxes()
        .render()
        .reorderable()
        .brushMode('1D-axes');

      // add hover event

      if (!wrapperElement) {
        return;
      }

      // if brushing, clear selected lines
      parcoord.current.on('brushend', () => {
        parcoord.current.unhighlight();
        closeContextMenu();
      });

      // Add event listeners just once
      wrapperElement.querySelector('#wrapper svg')?.addEventListener('mousemove', function mouseMoveHandler(ev: Event) {
        const { offsetX, offsetY } = ev as MouseEvent;
        highlightLineOnHover({ offsetX, offsetY });
      });

      wrapperElement.querySelector('#wrapper svg')?.addEventListener('mouseout', () => {
        setHoveredRunUuid('');
      });

      // rotate and truncate axis labels
      wrapperElement.querySelectorAll('.parcoords .label').forEach((e) => {
        const originalLabel = e.innerHTML;
        if (num_axes > axesRotateThreshold) {
          e.setAttribute('transform', 'rotate(-30)');
        }
        e.setAttribute('y', '-20');
        e.setAttribute('x', '20');
        const width_pre_truncation = e.getBoundingClientRect().width;
        if (width_pre_truncation > maxAxesLabelWidth) {
          e.innerHTML = truncateChartMetricString(originalLabel, axesLabelTruncationThreshold);
          if (originalLabel !== e.innerHTML) {
            attachCustomTooltip('axis-label-tooltip', originalLabel, e);
          }
        }
      });

      // truncate tick labels
      wrapperElement.querySelectorAll('.parcoords .tick text').forEach((e) => {
        const originalLabel = e.innerHTML;
        const width_pre_truncation = e.getBoundingClientRect().width;
        if (width_pre_truncation > maxTickLabelWidth) {
          e.innerHTML = truncateChartMetricString(originalLabel, tickLabelTruncationThreshold);
          if (originalLabel !== e.innerHTML) {
            attachCustomTooltip('tick-label-tooltip', originalLabel, e);
          }
        }
      });

      // draw color bar
      const stops = Array.from({ length: 10 }, (_, i) => i / 9);
      const lg = parcoord.current.svg
        .append('defs')
        .append('linearGradient')
        .attr('id', 'mygrad')
        .attr('x2', '0%')
        .attr('y1', '100%')
        .attr('y2', '0%'); // Vertical linear gradient

      stops.forEach((stop) => {
        lg.append('stop')
          .attr('offset', `${stop * 100}%`)
          .style('stop-color', color_set(minValue + stop * (maxValue - minValue)));
      });

      // place the color bar right after the last axis
      // D3's select() has a hard time inside shadow DOM, let's use querySelector instead
      const parcoord_dimensions = chartRef.current?.querySelector('svg')?.getBoundingClientRect();
      const last_axes = chartRef.current?.querySelector('.dimension:last-of-type');
      if (!last_axes) return;
      const last_axes_box = last_axes?.getBoundingClientRect();
      const last_axes_location = last_axes?.getAttribute('transform');
      // last_axes_location is a string like "transform(100)"
      if (!last_axes_location) return;
      const last_axes_location_value = parseFloat(last_axes_location.split('(')[1].split(')')[0]);
      if (parcoord_dimensions) {
        const rect = parcoord.current.svg.append('rect');
        rect
          .attr('x', last_axes_location_value + 20)
          .attr('y', 0)
          .attr('width', 20)
          .attr('height', last_axes_box.height - 40)
          .style('fill', 'url(#mygrad)');
      }
    }
  }, [
    // Don't retrigger this useEffect on the entire props object update, only
    // on the fields that are actually relevant
    data,
    chartWidth,
    chartHeight,
    selectedParams,
    selectedMetrics,
    onHover,
    axesRotateThreshold,
    highlightLineOnHover,
    chartRef,
    closeContextMenu,
  ]);

  return <div ref={chartRef} id="wrapper" style={{ width: props.width, height: props.height }} className="parcoords" />;
};

const ParallelCoordinatesPlot = (props: any) => {
  const wrapper = useRef<HTMLDivElement>(null);
  const { theme } = useDesignSystemTheme();

  const { layoutHeight, layoutWidth, setContainerDiv } = useDynamicPlotSize();

  const [isResizing, setIsResizing] = useState(true);
  const timeoutRef = useRef<ReturnType<typeof setTimeout>>();

  useEffect(() => {
    setContainerDiv(wrapper.current);
  }, [setContainerDiv]);

  useEffect(() => {
    setIsResizing(true);
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    timeoutRef.current = setTimeout(() => {
      setIsResizing(false);
    }, 300); // Unblock after 300 ms
  }, [layoutHeight, layoutWidth]);

  return (
    <div
      ref={wrapper}
      css={{
        overflow: 'hidden',
        flex: '1',
        paddingTop: '20px',
        fontSize: 0,
        '.parcoords': {
          backgroundColor: theme.colors.backgroundPrimary,
        },
        '.parcoords svg': {
          overflow: 'visible !important',
        },
        '.parcoords text.label': {
          fill: theme.colors.textPrimary,
        },
      }}
    >
      {isResizing ? (
        <RunsChartCardLoadingPlaceholder />
      ) : (
        <ParallelCoordinatesPlotImpl {...props} width={layoutWidth} height={layoutHeight} />
      )}
    </div>
  );
};

export default ParallelCoordinatesPlot;
