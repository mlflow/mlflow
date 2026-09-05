import { dump } from 'js-yaml';

export function formatJsonStringAsYaml(data: string): string {
  try {
    return dump(JSON.parse(data), { noRefs: true, lineWidth: 120, sortKeys: false }).trimEnd();
  } catch {
    return data;
  }
}
