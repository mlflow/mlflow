export function coerceToEnum<T extends Record<string, string>, K extends keyof T, V extends T[K] | undefined>(
  enumObj: T,
  value: any,
  fallback: V,
): V | T[keyof T] {
  if (value === undefined || value === null || typeof value !== 'string') {
    return fallback;
  }
  for (const v in enumObj) {
    if (enumObj[v] === value) return enumObj[v];
  }
  return fallback;
}
