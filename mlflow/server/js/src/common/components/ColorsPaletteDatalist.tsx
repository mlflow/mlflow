import { RUNS_COLOR_PALETTE } from '../color-palette';

export const COLORS_PALETTE_DATALIST_ID = 'mlflow_run_colors_select';

/**
 * Datalist containing design system colors palette, to be used by native color picker.
 */
export const ColorsPaletteDatalist = () => (
  <datalist id={COLORS_PALETTE_DATALIST_ID}>
    {RUNS_COLOR_PALETTE.map((color) => (
      <option key={color}>{color}</option>
    ))}
  </datalist>
);
