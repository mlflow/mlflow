import { useCallback } from 'react';
import { useLocalStorage } from '@databricks/web-shared/hooks';
import { TRACE_DENSITY_STORAGE_KEY_PREFIX } from '../utils/constants';

// Bump when the density options or default change so stale entries reset.
export const DENSITY_STORAGE_VERSION = 2;

/**
 * Row-height density for the V4 traces table. Maps to the DS `Table` `size` prop: `'default'` is the
 * standard row height, `'small'` is compact. The mock offers a third "Tall" option, but the DS Table
 * supports only these two sizes today, so density is a two-way choice.
 */
export type TracesV4Density = 'default' | 'small';

/** Standard rows by default for consistency with other MLflow tables. */
const DEFAULT_DENSITY: TracesV4Density = 'default';

export interface TracesV4DensityControl {
  density: TracesV4Density;
  setDensity: (density: TracesV4Density) => void;
}

/**
 * Persists the traces-v4 row-height preference in localStorage, scoped per experiment (mirroring the
 * column-visibility / sizing hooks). Synchronous read on mount avoids a density flip on first paint.
 */
export const useTracesV4Density = (experimentId: string): TracesV4DensityControl => {
  const [density, setStored] = useLocalStorage<TracesV4Density>({
    key: `${TRACE_DENSITY_STORAGE_KEY_PREFIX}.${experimentId}`,
    version: DENSITY_STORAGE_VERSION,
    initialValue: DEFAULT_DENSITY,
  });

  const setDensity = useCallback((next: TracesV4Density) => setStored(next), [setStored]);

  return { density, setDensity };
};
