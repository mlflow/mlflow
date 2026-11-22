import { dump, YAMLException } from 'js-yaml';

const OPTIONS = { noRefs: true, lineWidth: 120, sortKeys: false };
// eslint-disable-next-line no-control-regex -- js-yaml escapes these code points.
const NON_PRINTABLE_YAML = /[\x00-\x09\x0b-\x1f\x7f-\xa0\u2028\u2029\ud800-\udfff\ufeff\ufffe\uffff]/;
const PLAIN_SAFE_SCALAR = /^[a-z_][a-z0-9_-]*$/i;

const truncateYamlScalar = (value: string, limit: number, isKey: boolean): string | undefined => {
  if (NON_PRINTABLE_YAML.test(value)) return undefined;
  let prefix = value.slice(0, limit);
  if (PLAIN_SAFE_SCALAR.test(value)) return prefix;
  if (isKey) return undefined;

  const isStableLiteral = prefix.includes('\n') && !/(?:^|\n)[^ \n][^\n]{40}/.test(value);
  const isStableFolded = /(?:^|\n)[^ \n][^\n]{120}/.test(prefix);
  if (!isStableLiteral && !isStableFolded) return undefined;

  prefix = prefix.replace(/\n+$/, '');
  if (value.endsWith('\n\n')) return `${prefix}\n\n`;
  if (value.endsWith('\n')) return `${prefix}\n`;
  return prefix;
};

export function formatJsonStringAsYaml(data: string): string {
  try {
    return dump(JSON.parse(data), OPTIONS).trimEnd();
  } catch {
    return data;
  }
}

/** Dump only the content needed to fill a collapsed preview. */
export function getYamlCollectionPreview(
  data: string,
  maxLines: number,
  maxChars: number,
): { text: string; isComplete: boolean } | undefined {
  let value: unknown;
  try {
    value = JSON.parse(data);
  } catch {
    return undefined;
  }
  for (let budget = maxLines + 1; budget <= maxLines * 64; budget *= 2) {
    let remaining = budget;
    // Keep one formatter line beyond the visible prefix so only the discarded tail can differ.
    let remainingChars = maxChars + OPTIONS.lineWidth;
    let cut = false;
    let stringsCut = false;
    let unsafeScalar = false;
    const arrayCuts: unknown[][] = [];
    const objectCuts: Record<string, unknown>[] = [];
    const strings: string[] = [];
    const prune = (node: unknown): unknown => {
      if (typeof node === 'string') {
        strings.push(node);
        if (node.length > remainingChars) {
          const prefix = truncateYamlScalar(node, remainingChars, false);
          if (prefix === undefined) {
            unsafeScalar = true;
            return node;
          }
          stringsCut = true;
          remainingChars = 0;
          return prefix;
        }
        remainingChars -= node.length;
        return node;
      }
      if (Array.isArray(node)) {
        const items: unknown[] = [];
        for (const item of node) {
          if (remainingChars === 0 || remaining-- <= 0) {
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
          if (remainingChars === 0 || remaining-- <= 0) {
            cut = true;
            objectCuts.push(entries);
            break;
          }
          strings.push(key);
          let previewKey = key;
          let keyWasCut = false;
          if (key.length > remainingChars) {
            // js-yaml switches keys over 1024 characters to explicit-key syntax.
            const keyLimit = key.length > 1024 ? Math.max(remainingChars, 1025) : remainingChars;
            const prefix = truncateYamlScalar(key, keyLimit, true);
            if (prefix === undefined || Object.hasOwn(entries, prefix)) {
              unsafeScalar = true;
              return node;
            }
            stringsCut = true;
            keyWasCut = true;
            previewKey = prefix;
            remainingChars = 0;
          } else {
            remainingChars -= key.length;
          }
          entries[previewKey] = keyWasCut ? null : prune((node as Record<string, unknown>)[key]);
        }
        return entries;
      }
      return node;
    };
    try {
      const pruned = prune(value);
      if (unsafeScalar) break;
      // The cut marker must not be mistaken for an original key or scalar, even in escaped JSON.
      let marker = 'mlflow_yaml_preview_cut';
      const containsMarker = (candidate: string) => strings.some((text) => text.includes(candidate));
      for (let attempt = 0; containsMarker(marker); attempt++) {
        // Repeated marker-like scalar content must not make collision checks grow without a bound.
        if (attempt === 16) return { text: formatJsonStringAsYaml(data), isComplete: true };
        marker += '_';
      }
      for (const entries of objectCuts) entries[marker] = null;
      for (const items of arrayCuts) items[items.length - 1] = marker;
      const text = dump(pruned, OPTIONS).trimEnd();
      if (!cut) return { text, isComplete: !stringsCut };
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
  return { text: formatJsonStringAsYaml(data), isComplete: true };
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
export const yamlJsonMayContain = (data: string, filter: string): boolean => {
  if (!filter) return false;
  // Non-word filters may span formatting tokens; 2 may be a block scalar's indentation indicator.
  if (/[^a-z0-9_]/i.test(filter) || filter === '2') return true;
  let value: unknown;
  try {
    value = JSON.parse(data);
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
