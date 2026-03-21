import * as fs from 'fs';
import { XMLParser } from 'fast-xml-parser';
import fetch from 'node-fetch';

async function readSitemap(input: string): Promise<string> {
  if (/^https?:\/\//.test(input)) {
    const res = await fetch(input);
    if (!res.ok) {
      const responseBody = await res.text();
      throw new Error(`Failed to fetch ${input}: ${res.status} ${res.statusText}. Response body: ${responseBody}`);
    }
    return await res.text();
  } else {
    return await fs.promises.readFile(input, 'utf8');
  }
}

function normalizePath(url: string): string {
  const idx = url.indexOf('/latest/');
  return idx >= 0 ? url.slice(idx + 'latest/'.length) : url;
}

async function parseSitemap(input: string): Promise<Map<string, string>> {
  const xml = await readSitemap(input);
  const parser = new XMLParser();
  const parsed = parser.parse(xml);
  const urlset = parsed.urlset.url as { loc: string }[];

  const urlMap = new Map<string, string>();
  for (const { loc } of urlset) {
    const normalized = normalizePath(loc);
    urlMap.set(normalized, loc);
  }
  return urlMap;
}

function compareSitemaps(mapA: Map<string, string>, mapB: Map<string, string>) {
  const onlyInA: string[] = [];
  const onlyInB: string[] = [];
  const inBoth: string[] = [];

  for (const [url, _] of mapA) {
    if (!mapB.has(url)) {
      onlyInA.push(url);
    } else {
      inBoth.push(url);
    }
  }

  for (const url of mapB.keys()) {
    if (!mapA.has(url)) {
      onlyInB.push(url);
    }
  }

  return { onlyInA, onlyInB, inBoth };
}

(async () => {
  const fileA = process.argv[2];
  const fileB = process.argv[3];

  if (!fileA || !fileB) {
    console.error('Usage: tsx compare-sitemaps.ts <fileA|urlA> <fileB|urlB>');
    process.exit(1);
  }

  const [mapA, mapB] = await Promise.all([parseSitemap(fileA), parseSitemap(fileB)]);

  const { onlyInA, onlyInB, inBoth } = compareSitemaps(mapA, mapB);

  console.log(`URLs in both: ${inBoth.length}`);
  console.log(`Only in ${fileA}: ${onlyInA.length}`);
  onlyInA.forEach((url) => console.log(`  ${url}`));

  console.log(`Only in ${fileB}: ${onlyInB.length}`);
  onlyInB.forEach((url) => console.log(`  ${url}`));
})();
