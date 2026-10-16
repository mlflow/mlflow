/**
 * Attempt JSON.parse on a span attribute value. Returns the parsed value on
 * success, or the raw string when parsing fails (e.g. the value is already a
 * plain string rather than a JSON-encoded one).
 */
export function tryDeserializeAttribute(value: string): any {
  try {
    const parsed = JSON.parse(value);
    // Guard against precision loss for a top-level integer-literal value (e.g. an int64-range
    // ID like "2051281657916407550"): anything outside the safe-integer range has already been
    // rounded by `JSON.parse`, so keep the original string instead of the corrupted number.
    // Re-stringifying is not a sufficient check — JS prints the shortest text that round-trips
    // to the same double, so a rounded value like `1000000000000000100` (stored as
    // ...0128) reproduces its input text exactly. This matches the `Number.isSafeInteger`
    // policy in `decodeOtelAnyValue`. Only integer literals are checked — floats like "1.0" or
    // "1e-05" deserialize losslessly and are left alone. Numbers nested inside a JSON-encoded
    // attribute value (e.g. `mlflow.spanInputs`) are out of scope here and can still lose
    // precision. The text is trimmed before matching because `JSON.parse` accepts surrounding
    // whitespace; the untrimmed original is what gets returned.
    if (typeof parsed === 'number' && /^-?\d+$/.test(value.trim()) && !Number.isSafeInteger(parsed)) {
      return value;
    }
    return parsed;
  } catch (e) {
    return value;
  }
}
