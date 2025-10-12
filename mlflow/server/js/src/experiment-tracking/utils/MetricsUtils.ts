import type { MessageDescriptor } from 'react-intl';
import { defineMessages } from 'react-intl';
// @ts-expect-error TS(7016): Could not find a declaration file for module 'date... Remove this comment to see the full error message
import dateFormat from 'dateformat';
import { MLFLOW_SYSTEM_METRIC_PREFIX } from '../constants';

interface MetricHistoryEntry {
  key: string;
  value: string | number;
  step?: number;
  timestamp: number;
}

const nonNumericLabels = defineMessages({
  positiveInfinity: {
    defaultMessage: 'Positive infinity ({metricKey})',
    description: 'Label indicating positive infinity used as a hover text in a plot UI element',
  },

  positiveInfinitySymbol: {
    defaultMessage: '+∞',
    description: 'Label displaying positive infinity symbol displayed on a plot UI element',
  },

  negativeInfinity: {
    defaultMessage: 'Negative infinity ({metricKey})',
    description: 'Label indicating negative infinity used as a hover text in a plot UI element',
  },

  negativeInfinitySymbol: {
    defaultMessage: '-∞',
    description: 'Label displaying negative infinity symbol displayed on a plot UI element',
  },

  nan: {
    defaultMessage: 'Not a number ({metricKey})',
    description: 'Label indicating "not-a-number" used as a hover text in a plot UI element',
  },

  nanSymbol: {
    defaultMessage: 'NaN',
    description: 'Label displaying "not-a-number" symbol displayed on a plot UI element',
  },
});

/**
 * Function checks if a given value is maximum/minimum possible
 * 64-bit IEEE 754 number representation. If true, converts it to a JS-recognized
 * infinity value.
 */
export const normalizeInfinity = (value: number | string) => {
  const normalizedValue = typeof value === 'string' ? parseFloat(value) : value;
  if (normalizedValue === Number.MAX_VALUE) return Number.POSITIVE_INFINITY;
  if (normalizedValue === -Number.MAX_VALUE) return Number.NEGATIVE_INFINITY;
  return normalizedValue;
};

/**
 * We consider maximum IEEE 754 values (1.79...e+308) as "infinity-like" values
 * due to the backend's behavior and existing API contract. This function maps all
 * extermal ("infinity-like") metric values to NaN values so they will be
 * displayed correctly on plot.
 */
export const normalizeMetricsHistoryEntry = ({ key, timestamp, value, step }: MetricHistoryEntry) => ({
  key: key,
  value: normalizeInfinity(value),
  step: step || 0,
  timestamp: timestamp,
});

/**
 * Compares two numbers, being able to return "true" if
 * two NaN values are given.
 */
const numberEqual = (n1: number, n2: number) => {
  if (isNaN(n1)) {
    return isNaN(n2);
  }
  return n1 === n2;
};

/**
 * Finds multiple continuous sequences of a given number in the array.
 *
 * @example
 * findNumberChunks([1, 2, 4, 4, 5, 6], 4);
 * -> [ {startIndex: 2, endIndex: 3} ]
 * findNumberChunks([1, 2, 2, 9, 2, 2, 8], 2)
 * -> [ {startIndex: 1, endIndex: 2}, {startIndex: 4, endIndex: 5} ]
 */
export const findNumberChunks = <T extends number = number>(
  arr: T[],
  needle: T,
  indexDelta = 0,
): {
  startIndex: number;
  endIndex: number;
}[] => {
  if (arr.length < 1) {
    return [];
  }

  const startIndex = arr.findIndex((value) => numberEqual(needle, value));
  if (startIndex === -1) {
    return [];
  }
  let endIndex = startIndex;
  let i = startIndex + 1;
  while (i < arr.length) {
    if (numberEqual(needle, arr[i])) {
      endIndex = i;
    } else {
      break;
    }
    i++;
  }

  return [
    {
      startIndex: startIndex + indexDelta,
      endIndex: endIndex + indexDelta,
    },
    ...findNumberChunks(arr.slice(endIndex + 1), needle, endIndex + 1 + indexDelta),
  ];
};

/**
 * Clamps the index to the given array
 */
export const clampIndex = <T>(arr: T[], index: number) => {
  return index < 0 ? 0 : index > arr.length - 1 ? arr.length - 1 : index;
};

const INFINITY_LINE_TEMPLATE = {
  type: 'line',
  ysizemode: 'pixel',
  y0: 0,
  line: {
    width: 1,
    color: 'red',
  },
};

/**
 * Given starting/ending indices of a chunk and an array, it calculates
 * averaged "x" position. Used for placing annotations in the middle of their
 * ranges.
 */
export const getAveragedPositionOnXAxis = (
  chunk: {
    startIndex: number;
    endIndex: number;
  },
  xValues: (number | string)[],
) => {
  const startIndex = clampIndex(xValues, chunk.startIndex - 1);
  const endIndex = clampIndex(xValues, chunk.endIndex + 1);

  // If it's not a number, then it's a date!
  if (typeof xValues[0] === 'string') {
    const date1 = xValues[startIndex];
    const date2 = xValues[endIndex];

    const d1msecs = new Date(date1).getTime(); // get milliseconds
    const d2msecs = new Date(date2).getTime(); // get milliseconds

    const avgTime = (d1msecs + d2msecs) / 2;

    return dateFormat(new Date(avgTime), 'yyyy-mm-dd HH:MM:ss.l');
  }

  return ((xValues[startIndex] as number) + (xValues[endIndex] as number)) / 2;
};

/**
 * Internal helper method - creates plotly annotation for an infinity
 */
const createInfinityAnnotation = (label: string, text: string, x: number, y: number) => ({
  hoverlabel: { bgcolor: 'red' },
  align: 'center',
  yref: 'paper',
  yanchor: y === 0 ? 'bottom' : 'top',
  showarrow: false,
  font: {
    size: 15,
    color: 'red',
  },
  y,
  text,
  x,
  hovertext: label,
});

export interface GenerateInfinityAnnotationsProps {
  /**
   * Array containing x coordinates of the plot data
   */
  xValues: number[];
  /**
   * Array containing y coordinates of the plot data
   */
  yValues: number[];
  /**
   * `true` if data should be prepared for logarithmic scale
   */
  isLogScale?: boolean;
  /**
   * Function used for label translation
   */
  stringFormatter?: (value: MessageDescriptor | string) => string;
}

/**
 *
 * Ingests plot's x and y values and earches for infinity and NaN values, then
 * generates plotly.js-compatible shapes and annotations for  * a given metric.
 *
 * @param configuration see `GenerateInfinityAnnotationsProps` for configuration reference
 * @returns an object containing arrays of plotly.js annotations and shapes
 */
export const generateInfinityAnnotations = ({
  xValues,
  yValues,
  isLogScale = false,
  stringFormatter = (str) => str.toString(),
}: GenerateInfinityAnnotationsProps) => {
  const nanChunks = findNumberChunks(yValues, NaN);
  const posInfChunks = findNumberChunks(yValues, Number.POSITIVE_INFINITY);
  const negInfChunks = findNumberChunks(yValues, Number.NEGATIVE_INFINITY);

  const nanAnnotations = nanChunks.map((chunk) => ({
    hovertext: stringFormatter(nonNumericLabels.nan),
    text: stringFormatter(nonNumericLabels.nanSymbol),
    align: 'center',
    yanchor: 'bottom',
    showarrow: false,
    font: {
      size: 12,
      color: 'blue',
    },
    x: getAveragedPositionOnXAxis(chunk, xValues),
    y: 0,
  }));
  const posInfAnnotations = posInfChunks.map((chunk) =>
    createInfinityAnnotation(
      stringFormatter(nonNumericLabels.positiveInfinity),
      stringFormatter(nonNumericLabels.positiveInfinitySymbol),
      getAveragedPositionOnXAxis(chunk, xValues),
      1,
    ),
  );

  const negInfAnnotations = negInfChunks.map((chunk) =>
    createInfinityAnnotation(
      stringFormatter(nonNumericLabels.negativeInfinity),
      stringFormatter(nonNumericLabels.negativeInfinitySymbol),
      getAveragedPositionOnXAxis(chunk, xValues),
      0,
    ),
  );

  const nanShapes = nanChunks.map((chunk) => {
    return {
      fillcolor: 'blue',
      opacity: 0.1,
      yref: 'paper',
      y0: 0,
      y1: 1,
      line: {
        width: 0,
      },
      x0: xValues[clampIndex(xValues, chunk.startIndex - 1)],
      x1: xValues[clampIndex(xValues, chunk.endIndex + 1)],
    };
  });

  const posInfinityShapes = posInfChunks.reduce<any>((shapes, chunk) => {
    shapes.push({
      ...INFINITY_LINE_TEMPLATE,
      y1: 1000,
      x0: xValues[clampIndex(xValues, chunk.startIndex - 1)],
      x1: xValues[clampIndex(xValues, chunk.startIndex - 1)],
      yanchor: yValues[clampIndex(yValues, chunk.startIndex - 1)],
    });
    if (chunk.endIndex + 1 < xValues.length) {
      shapes.push({
        ...INFINITY_LINE_TEMPLATE,
        y1: 1000,
        x0: xValues[clampIndex(xValues, chunk.endIndex + 1)],
        x1: xValues[clampIndex(xValues, chunk.endIndex + 1)],
        yanchor: yValues[clampIndex(yValues, chunk.endIndex + 1)],
      });
    }
    return shapes;
  }, []);
  const negInfinityShapes = negInfChunks.reduce<any>((shapes, chunk) => {
    if (isLogScale) {
      shapes.push({
        ...INFINITY_LINE_TEMPLATE,
        ysizemode: 'scaled',
        y0: 0,
        y1: yValues[clampIndex(yValues, chunk.startIndex - 1)],
        yanchor: yValues[clampIndex(yValues, chunk.startIndex - 1)],

        x0: xValues[clampIndex(xValues, chunk.startIndex - 1)],
        x1: xValues[clampIndex(xValues, chunk.startIndex - 1)],
      });
      shapes.push({
        ...INFINITY_LINE_TEMPLATE,
        ysizemode: 'scaled',
        y0: 0,
        y1: yValues[clampIndex(yValues, chunk.endIndex + 1)],
        yanchor: yValues[clampIndex(yValues, chunk.endIndex + 1)],

        x0: xValues[clampIndex(xValues, chunk.endIndex + 1)],
        x1: xValues[clampIndex(xValues, chunk.endIndex + 1)],
      });
    } else {
      shapes.push({
        ...INFINITY_LINE_TEMPLATE,
        y1: -1000,
        x0: xValues[clampIndex(xValues, chunk.startIndex - 1)],
        x1: xValues[clampIndex(xValues, chunk.startIndex - 1)],
        yanchor: yValues[clampIndex(yValues, chunk.startIndex - 1)],
      });
      shapes.push({
        ...INFINITY_LINE_TEMPLATE,
        y1: -1000,
        x0: xValues[clampIndex(xValues, chunk.endIndex + 1)],
        x1: xValues[clampIndex(xValues, chunk.endIndex + 1)],
        yanchor: yValues[clampIndex(yValues, chunk.endIndex + 1)],
      });
    }
    return shapes;
  }, []);
  return {
    shapes: [...nanShapes, ...negInfinityShapes, ...posInfinityShapes],
    annotations: [...nanAnnotations, ...posInfAnnotations, ...negInfAnnotations],
  };
};

export const truncateChartMetricString = (fullStr: string, strLen: number) => {
  if (fullStr.length <= strLen) return fullStr;

  const separator = '...';

  const sepLen = separator.length,
    charsToShow = strLen - sepLen,
    frontChars = Math.ceil(charsToShow / 2),
    backChars = Math.floor(charsToShow / 2);

  return fullStr.substr(0, frontChars) + separator + fullStr.substr(fullStr.length - backChars);
};

const systemMetricPrefix = new RegExp(`^${MLFLOW_SYSTEM_METRIC_PREFIX}`);

export const isSystemMetricKey = (metricKey: string) => metricKey.match(systemMetricPrefix);

export const EXPERIMENT_RUNS_SAMPLE_METRIC_AUTO_REFRESH_INTERVAL = 30000;
