import { useState, useCallback } from 'react';

export type CostDimension = 'model' | 'provider' | 'caller';

const DIMENSION_CYCLE: CostDimension[] = ['model', 'provider', 'caller'];

export function useTraceCostDimension(defaultDimension: CostDimension = 'model') {
  const [dimension, setDimension] = useState<CostDimension>(defaultDimension);

  const toggleDimension = useCallback(() => {
    setDimension((prev) => {
      const idx = DIMENSION_CYCLE.indexOf(prev);
      return DIMENSION_CYCLE[(idx + 1) % DIMENSION_CYCLE.length];
    });
  }, []);

  return { dimension, setDimension, toggleDimension };
}
