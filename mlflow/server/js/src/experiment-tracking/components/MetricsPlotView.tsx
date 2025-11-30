/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */
import { MAX_LINE_SMOOTHNESS } from './MetricsPlotControls';

const EMA_THRESHOLD = 1;

export const EMA = (mArray: any, smoothingWeight: any) => {
  // If all elements in the set of metric values are constant, or if
  // the degree of smoothing is set to the minimum value, return the
  // original set of metric values
  if (smoothingWeight <= 1 || !mArray || mArray.length <= EMA_THRESHOLD || mArray.every((v: any) => v === mArray[0])) {
    return mArray;
  }

  const smoothness = smoothingWeight / (MAX_LINE_SMOOTHNESS + 1);
  const smoothedArray = [];
  let biasedElement = 0;
  for (let i = 0; i < mArray.length; i++) {
    if (!isNaN(mArray[i])) {
      biasedElement = biasedElement * smoothness + (1 - smoothness) * mArray[i];
      // To avoid biasing earlier elements toward smaller-than-accurate values, we divide
      // all elements by a `debiasedWeight` that asymptotically increases and approaches
      // 1 as the element index increases
      const debiasWeight = 1.0 - Math.pow(smoothness, i + 1);
      const debiasedElement = biasedElement / debiasWeight;
      smoothedArray.push(debiasedElement);
    } else {
      smoothedArray.push(mArray[i]);
    }
  }
  return smoothedArray;
};
