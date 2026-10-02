/**
 * Column-aware metric number formatting.
 *
 * The core idea: all values in a metric column share one format spec, derived
 * from the column's maximum absolute value. This keeps numbers visually comparable
 * column-by-column without needing to inspect exponents per row.
 *
 * Formatting rules:
 * - 4 significant figures, anchored to the largest value in the column
 * - Digit groups separated by spaces on both sides of the decimal point
 *   (e.g. "0.000 001 234", "1 234 567")
 * - For "normal" magnitudes (|exponent| ≤ 6): plain decimal with space grouping
 * - For extreme magnitudes: mantissa scaled to [1, 10) range, with a header
 *   annotation ("×10ⁿ") indicating the shared exponent
 * - Zero is always "0"
 * - Negative values get a leading "−"
 */

/** Groups an integer string from the right in threes: "1234567" → "1 234 567" */
function groupIntegerDigits(s: string): string {
  return s.replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
}

/** Groups a fractional digit string from the left in threes: "000001234" → "000 001 234" */
function groupFractionalDigits(s: string): string {
  return s.replace(/(\d{3})(?=\d)/g, '$1 ');
}

function formatNumber(abs: number, decimalPlaces: number): string {
  const fixed = abs.toFixed(decimalPlaces);
  const dotIndex = fixed.indexOf('.');
  if (dotIndex === -1) {
    return groupIntegerDigits(fixed);
  }
  const intPart = fixed.slice(0, dotIndex);
  const fracPart = fixed.slice(dotIndex + 1);
  return groupIntegerDigits(intPart) + '.' + groupFractionalDigits(fracPart);
}

function toSuperscript(n: number): string {
  const map: Record<string, string> = {
    '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
    '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹', '-': '⁻',
  };
  return String(n).split('').map((c) => map[c] ?? c).join('');
}

export interface ColumnFormatSpec {
  /** Format a single cell value for display. Returns '' for null/undefined. */
  format: (value: number | undefined | null) => string;
  /**
   * If set, render this string below the column name in the header.
   * Indicates a shared scale factor, e.g. "×10⁻⁶".
   */
  headerAnnotation?: string;
}

/**
 * Compute a format spec for a metric column from all its values.
 *
 * Pass the full list of values (including undefined/null for runs that don't
 * have the metric). The returned spec should be reused for every cell in the
 * column so all numbers are formatted consistently.
 */
export function computeColumnFormatSpec(values: (number | undefined | null)[]): ColumnFormatSpec {
  const finite = values.filter((v): v is number => typeof v === 'number' && isFinite(v));

  // Fallback for empty / all-undefined columns
  if (finite.length === 0) {
    return {
      format: (v) => (v != null ? String(v) : ''),
    };
  }

  const maxAbs = Math.max(...finite.map(Math.abs));

  if (maxAbs === 0) {
    return { format: (v) => (v != null ? '0' : '') };
  }

  const exponent = Math.floor(Math.log10(maxAbs));

  if (exponent >= -4 && exponent <= 4) {
    // Normal magnitude: fixed decimal, same number of places across the column.
    // decimalPlaces chosen so the largest value shows 4 significant figures.
    let decimalPlaces = Math.max(0, 3 - exponent);

    // Drop to zero decimal places only if no value in the column actually has
    // a fractional part at this precision — i.e. every value rounds to itself
    // with zero decimals. This correctly handles zeros and whole-number columns.
    if (decimalPlaces > 0 && finite.every((v) => parseFloat(v.toFixed(decimalPlaces)) === Math.round(v))) {
      decimalPlaces = 0;
    }

    return {
      format: (v) => {
        if (v == null) return '';
        if (!isFinite(v)) return String(v);
        const formatted = formatNumber(Math.abs(v), decimalPlaces);
        return v < 0 ? '-' + formatted : formatted;
      },
    };
  }

  // Extreme magnitude: scale to [1, 10) and annotate the header.
  const scale = Math.pow(10, exponent);
  const headerAnnotation = '×10' + toSuperscript(exponent);
  return {
    format: (v) => {
      if (v == null) return '';
      if (v === 0) return '0'; // zero has no magnitude to scale, always display as-is
      if (!isFinite(v)) return String(v);
      const scaled = Math.abs(v) / scale;
      const formatted = formatNumber(scaled, 3);
      return v < 0 ? '-' + formatted : formatted;
    },
    headerAnnotation,
  };
}
