import { useCallback, useMemo, useState } from 'react';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { shouldEnableNewEvalRunsUi } from '@mlflow/mlflow/src/common/utils/FeatureUtils';
import { classifyAssessmentValueKinds, type AssessmentValueKind } from '../utils/assessmentColumns';

const NO_KEYS: ReadonlySet<string> = new Set();

interface KnownKinds {
  scopeKey: string;
  kinds: ReadonlyMap<string, AssessmentValueKind>;
}

/**
 * The feedback and expectation names known to hold numbers (keys from `assessmentValueKindKey`), so a
 * filter on them can compare as numbers. The filter is built before its own results load, so the
 * caller reports each loaded page through `recordTraces` and kinds are remembered across the pages
 * loaded for `scopeKey`: a page that happens to lack an assessment doesn't change how it's compiled,
 * and an assessment seen with any non-numeric value stays text. Only tracked with the new
 * evaluation runs UI.
 */
export const useNumericAssessmentKeys = (scopeKey: string) => {
  const [known, setKnown] = useState<KnownKinds>({ scopeKey, kinds: new Map() });

  const recordTraces = useCallback(
    (traces: ModelTraceInfoV3[]) => {
      if (!shouldEnableNewEvalRunsUi()) {
        return;
      }
      const pageKinds = classifyAssessmentValueKinds(traces);
      setKnown((previous) => {
        const sameScope = previous.scopeKey === scopeKey;
        const next = new Map(sameScope ? previous.kinds : []);
        let changed = !sameScope;
        pageKinds.forEach((kind, key) => {
          const merged = next.get(key) === 'other' ? 'other' : kind;
          if (next.get(key) !== merged) {
            next.set(key, merged);
            changed = true;
          }
        });
        return changed ? { scopeKey, kinds: next } : previous;
      });
    },
    [scopeKey],
  );

  const numericAssessmentKeys = useMemo<ReadonlySet<string>>(() => {
    if (known.scopeKey !== scopeKey || known.kinds.size === 0) {
      return NO_KEYS;
    }
    return new Set([...known.kinds].filter(([, kind]) => kind === 'numeric').map(([key]) => key));
  }, [known, scopeKey]);

  return { numericAssessmentKeys, recordTraces };
};
