import markdownLinkCheck from 'markdown-link-check';
import { glob } from 'glob';
import * as fs from 'fs/promises';

let encounteredBrokenLinks = false;

async function main() {
  for (const filename of await glob('docs/**/*.mdx')) {
    console.log('[CHECKING FILE]', filename);

    const content = await fs.readFile(filename, 'utf8');
    const brokenLinks = await check(filename, content);

    if (brokenLinks.length > 0) {
      console.log('[BROKEN LINKS]');
      brokenLinks.forEach((result) => console.log(`${result.link} ${result.statusCode}`));
    } else {
      console.log('[NO BROKEN LINKS]');
    }
  }

  if (encounteredBrokenLinks) {
    console.error('Found some broken links!');
    process.exit(1);
  }
}

await main();

type Result = {
  link: string;
  status: 'alive' | 'dead' | 'ignored';
  statusCode: number;
  err: Error | null;
};

async function check(filename: string, content: string): Promise<Result[]> {
  return new Promise((resolve, reject) => {
    const config = {
      httpHeaders: [
        {
          urls: ['https://openai.com', 'https://platform.openai.com'],
          headers: {
            'User-Agent':
              'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.5 Safari/605.1.15',
          },
        },
      ],
      ignorePatterns: [
        { pattern: '^(?!http)' }, // internal links
        { pattern: '^http://127.0.0.1' }, // local dev
        { pattern: '^http://localhost' },
        { pattern: '^https://YOUR_DATABRICKS_HOST' },
      ],
    };

    markdownLinkCheck(content, config, function (err: Error | null, results: Result[]) {
      const brokenLinks = [];

      if (err) {
        console.error('Error', err);
        reject(err);
      } else {
        results.forEach(function (result: Result) {
          console.info(`[INFO] ${result.link} is ${result.status} ${result.statusCode}`);
          if (result.status === 'dead') {
            brokenLinks.push(result);
            encounteredBrokenLinks = true;
          }
        });

        resolve(brokenLinks);
      }
    });
  });
}
