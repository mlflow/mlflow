import { useState, useCallback } from 'react';

export type CostDimension = 'model' | 'provider';

export function useTraceCostDimension(defaultDimension: CostDimension = 'model') {
  const [dimension, setDimension] = useState<CostDimension>(defaultDimension);

  const toggleDimension = useCallback(() => {
    setDimension((prev) => (prev === 'model' ? 'provider' : 'model'));
  }, []);

  return { dimension, setDimension, toggleDimension };
}
