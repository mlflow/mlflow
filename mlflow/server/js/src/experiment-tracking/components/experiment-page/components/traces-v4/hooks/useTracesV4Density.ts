import { useCallback } from 'react';
import { useLocalStorage } from '@databricks/web-shared/hooks';
import { TRACE_DENSITY_STORAGE_KEY_PREFIX } from '../utils/constants';

// Bump when the density options or default change so stale entries reset.
export const DENSITY_STORAGE_VERSION = 3;

/**
 * Row-height density for the V4 traces table. The design-system table supplies the compact and
 * standard padding; Standard also allows input and output previews to use a readable second line.
 * Tall keeps standard table padding and allows longer input/output previews.
 */
export type TracesV4Density = 'small' | 'standard' | 'tall';

/** Compact rows by default — the trace table is dense and most users scan many rows at once. */
const DEFAULT_DENSITY: TracesV4Density = 'small';

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
