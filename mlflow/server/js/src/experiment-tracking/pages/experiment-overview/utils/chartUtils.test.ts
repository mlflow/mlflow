import { describe, it, expect } from '@jest/globals';
import { renderHook, act } from '@testing-library/react';
import { formatTimestampForTraceMetrics, getTimestampFromDataPoint, useChartZoom } from './chartUtils';
import type { MetricDataPoint } from '@databricks/web-shared/model-trace-explorer';

// Time intervals in seconds
const MINUTE_IN_SECONDS = 60;
const HOUR_IN_SECONDS = 3600;
const DAY_IN_SECONDS = 86400;
const MONTH_IN_SECONDS = 2592000;

describe('chartUtils', () => {
  describe('formatTimestampForTraceMetrics', () => {
    // Use a fixed timestamp for consistent testing: 2025-12-22 10:30:45 UTC
    const testTimestamp = new Date('2025-12-22T10:30:45Z').getTime();

    it('should format timestamp at minute level (timeInterval <= 60s)', () => {
      const result = formatTimestampForTraceMetrics(testTimestamp, MINUTE_IN_SECONDS);
      // Should show time only (hour:minute)
      expect(result).toMatch(/\d{1,2}:\d{2}/);
    });

    it('should format timestamp at hour level (timeInterval <= 3600s)', () => {
      const result = formatTimestampForTraceMetrics(testTimestamp, HOUR_IN_SECONDS);
      // Should show month/day and hour
      expect(result).toMatch(/\d{1,2}\/\d{1,2}/);
    });

    it('should format timestamp at day level (timeInterval <= 86400s)', () => {
      const result = formatTimestampForTraceMetrics(testTimestamp, DAY_IN_SECONDS);
      // Should show month/day
      expect(result).toMatch(/\d{1,2}\/\d{1,2}/);
    });

    it('should format timestamp at month level (timeInterval > 86400s)', () => {
      const result = formatTimestampForTraceMetrics(testTimestamp, MONTH_IN_SECONDS);
      // Should show short month and year (e.g., "Dec '25")
      expect(result).toMatch(/[A-Za-z]+.*\d{2}/);
    });

    it('should handle edge case at exactly 60 seconds (minute level)', () => {
      const result = formatTimestampForTraceMetrics(testTimestamp, 60);
      expect(result).toMatch(/\d{1,2}:\d{2}/);
    });

    it('should handle edge case at exactly 3600 seconds (hour level)', () => {
      const result = formatTimestampForTraceMetrics(testTimestamp, 3600);
      expect(result).toMatch(/\d{1,2}\/\d{1,2}/);
    });

    it('should handle edge case at exactly 86400 seconds (day level)', () => {
      const result = formatTimestampForTraceMetrics(testTimestamp, 86400);
      expect(result).toMatch(/\d{1,2}\/\d{1,2}/);
    });
  });

  describe('getTimestampFromDataPoint', () => {
    it('should extract timestamp from time_bucket dimension', () => {
      const dataPoint: MetricDataPoint = {
        metric_name: 'trace_count',
        dimensions: {
          time_bucket: '2025-12-22T10:00:00Z',
        },
        values: { COUNT: 42 },
      };

      const result = getTimestampFromDataPoint(dataPoint);
      expect(result).toBe(new Date('2025-12-22T10:00:00Z').getTime());
    });

    it('should return 0 when time_bucket is not present', () => {
      const dataPoint: MetricDataPoint = {
        metric_name: 'trace_count',
        dimensions: {
          status: 'OK',
        },
        values: { COUNT: 42 },
      };

      const result = getTimestampFromDataPoint(dataPoint);
      expect(result).toBe(0);
    });

    it('should return 0 when dimensions is empty', () => {
      const dataPoint: MetricDataPoint = {
        metric_name: 'trace_count',
        dimensions: {},
        values: { COUNT: 42 },
      };

      const result = getTimestampFromDataPoint(dataPoint);
      expect(result).toBe(0);
    });

    it('should handle data point with multiple dimensions including time_bucket', () => {
      const dataPoint: MetricDataPoint = {
        metric_name: 'latency',
        dimensions: {
          status: 'OK',
          time_bucket: '2025-12-22T15:30:00Z',
          name: 'my_trace',
        },
        values: { AVG: 150.5, P99: 234.5 },
      };

      const result = getTimestampFromDataPoint(dataPoint);
      expect(result).toBe(new Date('2025-12-22T15:30:00Z').getTime());
    });
  });

  describe('useChartZoom', () => {
    interface TestDataPoint {
      name: string;
      value: number;
    }

    const testData: TestDataPoint[] = [
      { name: '10:00', value: 42 },
      { name: '11:00', value: 58 },
      { name: '12:00', value: 100 },
      { name: '13:00', value: 75 },
      { name: '14:00', value: 90 },
    ];

    it('should initialize with full data and not zoomed', () => {
      const { result } = renderHook(() => useChartZoom(testData, 'name'));

      expect(result.current.zoomedData).toEqual(testData);
      expect(result.current.isZoomed).toBe(false);
      expect(result.current.refAreaLeft).toBeNull();
      expect(result.current.refAreaRight).toBeNull();
    });

    it('should zoom to selected range when mouse events complete a selection', () => {
      const { result } = renderHook(() => useChartZoom(testData, 'name'));

      // Simulate mouse down at 11:00
      act(() => {
        result.current.handleMouseDown({ activeLabel: '11:00' });
      });
      expect(result.current.refAreaLeft).toBe('11:00');

      // Simulate mouse move to 13:00
      act(() => {
        result.current.handleMouseMove({ activeLabel: '13:00' });
      });
      expect(result.current.refAreaRight).toBe('13:00');

      // Simulate mouse up to complete selection
      act(() => {
        result.current.handleMouseUp();
      });

      // Should now be zoomed to the selected range (11:00, 12:00, 13:00)
      expect(result.current.isZoomed).toBe(true);
      expect(result.current.zoomedData).toEqual([
        { name: '11:00', value: 58 },
        { name: '12:00', value: 100 },
        { name: '13:00', value: 75 },
      ]);
      // Selection area should be cleared after zoom
      expect(result.current.refAreaLeft).toBeNull();
      expect(result.current.refAreaRight).toBeNull();
    });

    it('should handle reverse selection (right to left)', () => {
      const { result } = renderHook(() => useChartZoom(testData, 'name'));

      // Select from 13:00 to 11:00 (reverse)
      act(() => {
        result.current.handleMouseDown({ activeLabel: '13:00' });
      });
      act(() => {
        result.current.handleMouseMove({ activeLabel: '11:00' });
      });
      act(() => {
        result.current.handleMouseUp();
      });

      // Should still zoom to correct range
      expect(result.current.isZoomed).toBe(true);
      expect(result.current.zoomedData).toHaveLength(3);
      expect(result.current.zoomedData[0].name).toBe('11:00');
      expect(result.current.zoomedData[2].name).toBe('13:00');
    });

    it('should reset zoom when zoomOut is called', () => {
      const { result } = renderHook(() => useChartZoom(testData, 'name'));

      // First zoom in
      act(() => {
        result.current.handleMouseDown({ activeLabel: '11:00' });
      });
      act(() => {
        result.current.handleMouseMove({ activeLabel: '13:00' });
      });
      act(() => {
        result.current.handleMouseUp();
      });

      expect(result.current.isZoomed).toBe(true);
      expect(result.current.zoomedData).toHaveLength(3);

      // Then zoom out
      act(() => {
        result.current.zoomOut();
      });

      expect(result.current.isZoomed).toBe(false);
      expect(result.current.zoomedData).toEqual(testData);
    });

    it('should reset zoom when data changes (e.g., time range changed)', () => {
      const { result, rerender } = renderHook(({ data }) => useChartZoom(data, 'name'), {
        initialProps: { data: testData },
      });

      // First zoom in
      act(() => {
        result.current.handleMouseDown({ activeLabel: '11:00' });
      });
      act(() => {
        result.current.handleMouseMove({ activeLabel: '13:00' });
      });
      act(() => {
        result.current.handleMouseUp();
      });

      expect(result.current.isZoomed).toBe(true);
      expect(result.current.zoomedData).toHaveLength(3);

      // Now simulate time range change by providing new data
      const newData: TestDataPoint[] = [
        { name: '08:00', value: 20 },
        { name: '09:00', value: 30 },
        { name: '10:00', value: 40 },
      ];

      rerender({ data: newData });

      // Zoom should be reset
      expect(result.current.isZoomed).toBe(false);
      expect(result.current.zoomedData).toEqual(newData);
    });

    it('should not zoom if selection is only one point', () => {
      const { result } = renderHook(() => useChartZoom(testData, 'name'));

      // Select same point for start and end
      act(() => {
        result.current.handleMouseDown({ activeLabel: '11:00' });
      });
      act(() => {
        result.current.handleMouseMove({ activeLabel: '11:00' });
      });
      act(() => {
        result.current.handleMouseUp();
      });

      // Should not zoom (need at least 2 points)
      expect(result.current.isZoomed).toBe(false);
      expect(result.current.zoomedData).toEqual(testData);
    });

    it('should not update refAreaRight if mouse move happens without mouse down', () => {
      const { result } = renderHook(() => useChartZoom(testData, 'name'));

      // Move without down
      act(() => {
        result.current.handleMouseMove({ activeLabel: '12:00' });
      });

      expect(result.current.refAreaLeft).toBeNull();
      expect(result.current.refAreaRight).toBeNull();
    });

    it('should clear selection on mouse up without valid selection', () => {
      const { result } = renderHook(() => useChartZoom(testData, 'name'));

      // Mouse up without any selection
      act(() => {
        result.current.handleMouseUp();
      });

      expect(result.current.isZoomed).toBe(false);
      expect(result.current.refAreaLeft).toBeNull();
      expect(result.current.refAreaRight).toBeNull();
    });
  });
});
