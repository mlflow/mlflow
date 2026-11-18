import { dump } from 'js-yaml';

const OPTIONS = { noRefs: true, lineWidth: 120, sortKeys: false };

export function formatJsonStringAsYaml(data: string): string {
  try {
    return dump(JSON.parse(data), OPTIONS).trimEnd();
  } catch {
    return data;
  }
}

/** Dump only the collection entries needed to fill a collapsed preview. */
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
        if (attempt === 16) return { text: formatJsonStringAsYaml(data), isComplete: true };
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
    } catch {
      break;
    }
  }
  return { text: formatJsonStringAsYaml(data), isComplete: true };
}
