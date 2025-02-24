import {
  clampIndex,
  findNumberChunks,
  generateInfinityAnnotations,
  getAveragedPositionOnXAxis,
  normalizeInfinity,
  normalizeMetricsHistoryEntry,
} from './MetricsUtils';
import Utils from '../../common/utils/Utils';

jest.mock('../../common/utils/Utils');

describe('MetricsUtils', () => {
  test('isExtremeFloat', () => {
    expect(normalizeInfinity(0)).toBe(0);
    expect(normalizeInfinity(123)).toBe(123);
    expect(normalizeInfinity(1230000e10)).toBe(1230000e10);
    expect(normalizeInfinity(NaN)).toBe(NaN);
    expect(normalizeInfinity(Number.MAX_VALUE)).toBe(Number.POSITIVE_INFINITY);
    expect(normalizeInfinity(-Number.MAX_VALUE)).toBe(Number.NEGATIVE_INFINITY);
    expect(normalizeInfinity(Number.MAX_VALUE.toString())).toBe(Number.POSITIVE_INFINITY);
    expect(normalizeInfinity((-Number.MAX_VALUE).toString())).toBe(Number.NEGATIVE_INFINITY);
  });
  test('normalizeMetricsHistoryEntry', () => {
    expect(normalizeMetricsHistoryEntry({ key: 'foobar', value: '123', timestamp: 123, step: 5 })).toEqual({
      key: 'foobar',
      value: 123,
      timestamp: 123,
      step: 5,
    });

    expect(normalizeMetricsHistoryEntry({ key: 'foobar', value: Number.MAX_VALUE, timestamp: 123 })).toEqual({
      key: 'foobar',
      value: Number.POSITIVE_INFINITY,
      timestamp: 123,
      step: 0,
    });
  });
  test('findNumberChunks', () => {
    expect(findNumberChunks([1, 2, 4, 4, 5, 6], 4)).toEqual([{ startIndex: 2, endIndex: 3 }]);
    expect(findNumberChunks([1, 2, 2, 9, 2, 2, 8], 2)).toEqual([
      { startIndex: 1, endIndex: 2 },
      { startIndex: 4, endIndex: 5 },
    ]);

    expect(findNumberChunks([NaN, NaN, NaN, 9, NaN, NaN, NaN], NaN)).toEqual([
      { startIndex: 0, endIndex: 2 },
      { startIndex: 4, endIndex: 6 },
    ]);

    expect(findNumberChunks([NaN, NaN, NaN], NaN)).toEqual([{ startIndex: 0, endIndex: 2 }]);
    expect(
      findNumberChunks([Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, 1, 2, 3], Number.POSITIVE_INFINITY),
    ).toEqual([{ startIndex: 0, endIndex: 1 }]);

    expect(
      findNumberChunks(
        [
          Number.POSITIVE_INFINITY,
          Number.NEGATIVE_INFINITY,
          Number.POSITIVE_INFINITY,
          Number.NEGATIVE_INFINITY,
          Number.POSITIVE_INFINITY,
        ],
        Number.POSITIVE_INFINITY,
      ),
    ).toEqual([
      { startIndex: 0, endIndex: 0 },
      { startIndex: 2, endIndex: 2 },
      { startIndex: 4, endIndex: 4 },
    ]);

    expect(
      findNumberChunks(
        [
          Number.POSITIVE_INFINITY,
          Number.NEGATIVE_INFINITY,
          Number.POSITIVE_INFINITY,
          Number.NEGATIVE_INFINITY,
          Number.POSITIVE_INFINITY,
        ],
        Number.NEGATIVE_INFINITY,
      ),
    ).toEqual([
      { startIndex: 1, endIndex: 1 },
      { startIndex: 3, endIndex: 3 },
    ]);
  });

  test('clampIndex', () => {
    expect(clampIndex([1, 1, 1, 1, 1], 0)).toEqual(0);
    expect(clampIndex([1, 1, 1, 1, 1], -1)).toEqual(0);
    expect(clampIndex([1, 1, 1, 1, 1], 3)).toEqual(3);
    expect(clampIndex([1, 1, 1, 1, 1], 4)).toEqual(4);
    expect(clampIndex([1, 1, 1, 1, 1], 5)).toEqual(4);
  });

  test('getAveragedPositionOnXAxis', () => {
    expect(getAveragedPositionOnXAxis({ startIndex: 1, endIndex: 1 }, [1, 2, 3, 4, 5])).toEqual(2);
    expect(getAveragedPositionOnXAxis({ startIndex: 1, endIndex: 2 }, [1, 2, 3, 4, 5])).toEqual((2 + 3) / 2);
    expect(getAveragedPositionOnXAxis({ startIndex: 0, endIndex: 2 }, [1, 2, 3, 4, 5])).toEqual((2 + 3) / 2);
    expect(getAveragedPositionOnXAxis({ startIndex: 1, endIndex: 3 }, [1, 2, 3, 4, 5])).toEqual(3);
    expect(getAveragedPositionOnXAxis({ startIndex: 0, endIndex: 4 }, [1, 2, 3, 4, 5])).toEqual(3);

    getAveragedPositionOnXAxis({ startIndex: 1, endIndex: 2 }, [
      '2020-01-01 02:00:00',
      '2020-01-02 02:00:00',
      '2020-01-03 02:00:00',
      '2020-01-04 02:00:00',
      '2020-01-05 02:00:00',
    ]);
  });

  describe('generateInfinityAnnotations', () => {
    const xValuesNumber = [0, 1, 2, 3, 4, 5, 6, 7];
    test('positive and negative infinity annotations', () => {
      const yValues = [
        1,
        2,
        Number.POSITIVE_INFINITY,
        Number.POSITIVE_INFINITY,
        8,
        Number.NEGATIVE_INFINITY,
        Number.NEGATIVE_INFINITY,
        9,
      ];
      const result = generateInfinityAnnotations({ xValues: xValuesNumber, yValues });

      expect(result.annotations).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            x: (2 + 3) / 2,
            y: 1,
            yanchor: 'top',
            yref: 'paper',
          }),
          expect.objectContaining({
            x: (5 + 6) / 2,
            y: 0,
            yanchor: 'bottom',
            yref: 'paper',
          }),
        ]),
      );

      expect(result.shapes).toEqual(
        expect.arrayContaining([
          // positive infinity shapes:
          expect.objectContaining({ y0: 0, y1: 1000, yanchor: 2, x0: 1, x1: 1 }),
          expect.objectContaining({ y0: 0, y1: 1000, yanchor: 8, x0: 4, x1: 4 }),
          // negative infinity shapes:
          expect.objectContaining({ y0: 0, y1: -1000, yanchor: 8, x0: 4, x1: 4 }),
          expect.objectContaining({ y0: 0, y1: -1000, yanchor: 9, x0: 7, x1: 7 }),
        ]),
      );
    });

    test('infinity annotations on edges', () => {
      const yValues = [
        Number.POSITIVE_INFINITY,
        Number.POSITIVE_INFINITY,
        Number.POSITIVE_INFINITY,
        Number.POSITIVE_INFINITY,
        8,
        7,
        Number.NEGATIVE_INFINITY,
        Number.NEGATIVE_INFINITY,
      ];
      const result = generateInfinityAnnotations({ xValues: xValuesNumber, yValues });

      expect(result.annotations).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            x: 2,
            y: 1,
            yanchor: 'top',
            yref: 'paper',
          }),
        ]),
      );

      expect(result.shapes).toEqual(
        expect.arrayContaining([
          // positive infinity shapes:
          expect.objectContaining({
            y0: 0,
            y1: 1000,
            yanchor: Number.POSITIVE_INFINITY,
            x0: 0,
            x1: 0,
          }),
          expect.objectContaining({ y0: 0, y1: 1000, yanchor: 8, x0: 4, x1: 4 }),
          // negative infinity shapes:
          expect.objectContaining({ y0: 0, y1: -1000, yanchor: 7, x0: 5, x1: 5 }),
          expect.objectContaining({
            y0: 0,
            y1: -1000,
            yanchor: Number.NEGATIVE_INFINITY,
            x0: 7,
            x1: 7,
          }),
        ]),
      );
    });

    test('infinity annotations on log scale', () => {
      const yValues = [
        1,
        Number.POSITIVE_INFINITY,
        Number.POSITIVE_INFINITY,
        4,
        8,
        Number.NEGATIVE_INFINITY,
        Number.NEGATIVE_INFINITY,
        4,
      ];
      const result = generateInfinityAnnotations({
        xValues: xValuesNumber,
        yValues,
        isLogScale: true,
      });

      expect(result.annotations).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            x: 1.5,
            y: 1,
            yanchor: 'top',
            yref: 'paper',
          }),
        ]),
      );

      expect(result.shapes).toEqual(
        expect.arrayContaining([
          // positive infinity shapes:
          expect.objectContaining({
            y0: 0,
            y1: 1000,
            yanchor: 1,
            x0: 0,
            x1: 0,
          }),
          expect.objectContaining({
            y0: 0,
            y1: 1000,
            yanchor: 4,
            x0: 3,
            x1: 3,
          }),

          // negative infinity shapes:
          expect.objectContaining({
            y0: 0,
            y1: 8,
            yanchor: 8,
            x0: 4,
            x1: 4,
          }),
          expect.objectContaining({
            y0: 0,
            y1: 4,
            yanchor: 4,
            x0: 7,
            x1: 7,
          }),
        ]),
      );
    });
  });
});
