import { dump, YAMLException } from 'js-yaml';

const OPTIONS = { noRefs: true, lineWidth: 120, sortKeys: false };

export function formatJsonStringAsYaml(data: string, parsedData?: unknown): string {
  try {
    return dump(parsedData === undefined ? JSON.parse(data) : parsedData, OPTIONS).trimEnd();
  } catch {
    return data;
  }
}

/** Dump only the collection entries needed to fill a collapsed preview. */
export function getYamlCollectionPreview(
  data: string,
  maxLines: number,
  maxChars: number,
  parsedData?: unknown,
): { text: string; isComplete: boolean } | undefined {
  let value: unknown;
  try {
    value = parsedData === undefined ? JSON.parse(data) : parsedData;
  } catch {
    return undefined;
  }
  // shortcut: Scalar strings keep js-yaml's complete formatting; bound them once exact scalar-prefix formatting is available.
  for (let budget = maxLines + 1; budget <= maxLines * 64; budget *= 2) {
    let remaining = budget;
    let cut = false;
    const arrayCuts: unknown[][] = [];
    const objectCuts: Record<string, unknown>[] = [];
    const strings: string[] = [];
    const prune = (node: unknown): unknown => {
      if (typeof node === 'string') strings.push(node);
      if (Array.isArray(node)) {
        const items: unknown[] = [];
        for (const item of node) {
          if (remaining-- <= 0) {
            cut = true;
            items.push(null);
            arrayCuts.push(items);
            break;
          }
          items.push(prune(item));
        }
        return items;
      }
      if (node !== null && typeof node === 'object') {
        const entries: Record<string, unknown> = Object.create(null);
        for (const key in node) {
          if (!Object.hasOwn(node, key)) continue;
          if (remaining-- <= 0) {
            cut = true;
            objectCuts.push(entries);
            break;
          }
          strings.push(key);
          entries[key] = prune((node as Record<string, unknown>)[key]);
        }
        return entries;
      }
      return node;
    };
    try {
      const pruned = prune(value);
      // The cut marker must not be mistaken for an original key or scalar, even in escaped JSON.
      let marker = 'mlflow_yaml_preview_cut';
      const containsMarker = (candidate: string) => strings.some((text) => text.includes(candidate));
      for (let attempt = 0; containsMarker(marker); attempt++) {
        // Repeated marker-like scalar content must not make collision checks grow without a bound.
        if (attempt === 16) return { text: formatJsonStringAsYaml(data, value), isComplete: true };
        marker += '_';
      }
      for (const entries of objectCuts) entries[marker] = null;
      for (const items of arrayCuts) items[items.length - 1] = marker;
      const text = dump(pruned, OPTIONS).trimEnd();
      if (!cut) return { text, isComplete: true };
      const lineStart = text.lastIndexOf('\n', text.indexOf(marker));
      const prefix = lineStart === -1 ? '' : text.slice(0, lineStart);
      let lines = 1;
      for (let index = 0; index < prefix.length && lines < maxLines; index++) {
        if (prefix[index] === '\n') lines++;
      }
      if (prefix.length >= maxChars || lines >= maxLines) return { text: prefix, isComplete: false };
    } catch (error) {
      // Preserve the formatter fallback for unsupported YAML or native nesting/string limits.
      if (!(error instanceof YAMLException) && !(error instanceof RangeError)) throw error;
      break;
    }
  }
  return { text: formatJsonStringAsYaml(data, value), isComplete: true };
}

const YAML_ESCAPES = new Map(
  Object.entries({
    '\0': '\\0',
    '\x07': '\\a',
    '\b': '\\b',
    '\t': '\\t',
    '\n': '\\n',
    '\v': '\\v',
    '\f': '\\f',
    '\r': '\\r',
    '\x1b': '\\e',
    '\x85': '\\N',
    '\xa0': '\\_',
    '\u2028': '\\L',
    '\u2029': '\\P',
  }),
);

const mayContainYamlString = (value: string, filter: string): boolean => {
  if (value.toLowerCase().includes(filter)) return true;
  // js-yaml 3 also escapes astral characters; search their displayed form, including lone surrogates.
  const escaped = value.replace(
    // eslint-disable-next-line no-control-regex -- YAML escapes non-printable code points.
    /[\x00-\x1f\x7f-\xa0\u2028\u2029\ud800-\udfff\ufeff\ufffe\uffff\u{10000}-\u{10ffff}]/gu,
    (character) => {
      const simple = YAML_ESCAPES.get(character);
      if (simple) return simple;
      const code = character.codePointAt(0) ?? 0;
      const [prefix, width] = code <= 0xff ? ['x', 2] : code <= 0xffff ? ['u', 4] : ['U', 8];
      return '\\' + prefix + code.toString(16).toUpperCase().padStart(width, '0');
    },
  );
  return escaped.toLowerCase().includes(filter);
};

/** Reject impossible word searches without dumping the collection; uncertain YAML syntax stays searchable. */
export const yamlJsonMayContain = (data: string, filter: string, parsedData?: unknown): boolean => {
  if (!filter) return false;
  // Non-word filters may span formatting tokens; 2 may be a block scalar's indentation indicator.
  if (/[^a-z0-9_]/i.test(filter) || filter === '2') return true;
  let value: unknown;
  try {
    value = parsedData === undefined ? JSON.parse(data) : parsedData;
  } catch {
    return true;
  }
  const pending = [value];
  while (pending.length) {
    const node = pending.pop();
    if (typeof node === 'string') {
      if (mayContainYamlString(node, filter)) return true;
    } else if (node !== null && typeof node === 'object') {
      for (const key in node) {
        if (!Object.hasOwn(node, key)) continue;
        if (!Array.isArray(node) && mayContainYamlString(key, filter)) return true;
        pending.push((node as Record<string, unknown>)[key]);
      }
    } else if (typeof node === 'number' && !Number.isFinite(node)) {
      return true;
    } else if (String(node).toLowerCase().includes(filter)) {
      return true;
    }
  }
  return false;
};
