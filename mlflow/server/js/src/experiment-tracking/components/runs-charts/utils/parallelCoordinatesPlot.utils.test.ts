import type { RunsChartsRunData } from '../components/RunsCharts.common';
import type { ParallelCoordinateDataEntry } from './parallelCoordinatesPlot.utils';
import { PARALLEL_CHART_MAX_NUMBER_STRINGS } from './parallelCoordinatesPlot.utils';
import { filterParallelCoordinateData } from './parallelCoordinatesPlot.utils';
import { processParallelCoordinateData } from './parallelCoordinatesPlot.utils';

describe('ParallelCoordinatesPlot utilities', () => {
  describe('filterParallelCoordinateData', () => {
    test('filters out NaNs and nulls', () => {
      const data: ParallelCoordinateDataEntry[] = [];

      for (let i = 0; i < 100; i++) {
        data.push({
          uuid: i,
          left: Math.random(),
          right: Math.random(),
        });
      }
      data.push({
        uuid: 100,
        left: NaN,
        right: Math.random(),
      });
      data.push({
        uuid: 101,
        left: null,
        right: Math.random(),
      });

      expect(data.length).toBe(102);
      const filteredData = filterParallelCoordinateData(data, ['left', 'right']);
      expect(filteredData.length).toBe(100);
    });

    test('keep a max of 30 unique string values', () => {
      const data = [];
      const divisor = 2;
      for (let i = 0; i < 100; i++) {
        data.push({
          uuid: i,
          left: `${Math.floor(i / divisor)}a`,
          right: Math.random(),
        });
      }
      expect(data.length).toBe(100);
      const filteredData = filterParallelCoordinateData(data, ['left', 'right']);
      expect(filteredData.length).toBe(PARALLEL_CHART_MAX_NUMBER_STRINGS * divisor);
    });

    test('displays 100 nums over 50 strings', () => {
      const data = [];

      for (let i = 0; i < 100; i++) {
        data.push({
          uuid: i,
          left: Math.random(),
          right: Math.random(),
        });
      }

      // "left" is populated with strings 50 times
      for (let i = 100; i < 150; i++) {
        data.push({
          uuid: i,
          left: `${Math.floor(i / 2)}a`,
          right: Math.random(),
        });
      }

      expect(data.length).toBe(150);
      const filteredData = filterParallelCoordinateData(data, ['left', 'right']);
      expect(filteredData.length).toBe(100);
    });

    test('displays 99 (effectively 90) strings over 51 numbers', () => {
      const data = [];
      const divisor = 3;
      // "left" is populated with numbers 50 times
      for (let i = 0; i < 51; i++) {
        data.push({
          uuid: i,
          left: Math.random(),
          right: Math.random(),
        });
      }

      // "left" is populated with strings 99 times
      for (let i = 51; i < 150; i++) {
        data.push({
          uuid: i,
          left: `${Math.floor(i / divisor)}a`,
          right: Math.random(),
        });
      }
      expect(data.length).toBe(150);
      const filteredData = filterParallelCoordinateData(data, ['left', 'right']);
      expect(filteredData.length).toBe(divisor * PARALLEL_CHART_MAX_NUMBER_STRINGS);
    });

    test('prepares data for 3 columns', () => {
      const data = [];
      const divisor = 4;
      for (let i = 0; i < 200; i++) {
        if (i % 4 === 0) {
          data.push({
            uuid: i,
            left: Math.random(),
            middle: 'a',
            right: Math.random(),
          });
        } else {
          data.push({
            uuid: i,
            left: `${Math.floor(i / divisor)}a`,
            middle: 'b',
            right: Math.random(),
          });
        }
      }

      expect(data.length).toBe(200);
      const filteredData = filterParallelCoordinateData(data, ['left', 'right', 'middle']);
      expect(filteredData.length).toBe((divisor - 1) * PARALLEL_CHART_MAX_NUMBER_STRINGS);
    });
  });

  describe('processParallelCoordinateData', () => {
    test('prepares parallel coordinates data entries and keeps all runs when metrics and params are matching', () => {
      const data: RunsChartsRunData[] = [];

      for (let i = 0; i < 20; i++) {
        data.push({
          uuid: i.toString(),
          params: {
            param_1: { key: 'param_1', value: 'abc' },
          },
          metrics: {
            metric_1: { key: 'metric_1', value: Math.random(), timestamp: 0, step: 0 },
            // Some extraneous metric
            metric_2: { key: 'metric_2', value: Math.random(), timestamp: 0, step: 0 },
          },
        } as any);
      }

      const filteredData = processParallelCoordinateData(data, ['param_1'], ['metric_1']);
      expect(filteredData.length).toBe(data.length);
    });

    test('prepares parallel coordinates data entries and discards runs with incomplete data', () => {
      const data: RunsChartsRunData[] = [];

      for (let i = 0; i < 20; i++) {
        data.push({
          uuid: i.toString(),
          params: {
            param_1: { key: 'param_1', value: 'abc' },
          },
          metrics: {
            metric_2: { key: 'metric_2', value: Math.random(), timestamp: 0, step: 0 },
          },
        } as any);
      }
      for (let i = 20; i < 40; i++) {
        data.push({
          uuid: i.toString(),
          params: {
            param_1: { key: 'param_1', value: 'abc' },
          },
          metrics: {
            metric_1: { key: 'metric_1', value: Math.random(), timestamp: 0, step: 0 },
            metric_2: { key: 'metric_2', value: Math.random(), timestamp: 0, step: 0 },
          },
        } as any);
      }
      for (let i = 40; i < 60; i++) {
        data.push({
          uuid: i.toString(),
          params: {},
          metrics: {
            metric_1: { key: 'metric_1', value: Math.random(), timestamp: 0, step: 0 },
          },
        } as any);
      }

      expect(data.length).toBe(60);
      const filteredData = processParallelCoordinateData(data, ['param_1'], ['metric_1', 'metric_2']);
      expect(filteredData.length).toBe(20);
    });
  });
});
