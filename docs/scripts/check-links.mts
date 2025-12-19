import markdownLinkCheck from 'markdown-link-check';
import { glob } from 'glob';
import * as fs from 'fs/promises';

async function main() {
  let encounteredBrokenLinks = false;

  const checkExternalLinks = process.env.CHECK_EXTERNAL_LINKS === 'true';
  for (const filename of await glob('docs/**/*.mdx')) {
    console.log('[CHECKING FILE]', filename);

    const content = await fs.readFile(filename, 'utf8');
    const brokenLinks = await check(content, checkExternalLinks);

    if (brokenLinks.length > 0) {
      console.log('[BROKEN LINKS]');
      brokenLinks.forEach((result) => console.log(`${result.link} ${result.statusCode}`));
      encounteredBrokenLinks = true;
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

async function check(content: string, checkExternalLinks: boolean): Promise<Result[]> {
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
      ignorePatterns: checkExternalLinks
        ? [
            { pattern: '^(?!http)' }, // relative links
            { pattern: '^http:\/\/127\.0\.0\.1' }, // local dev
            { pattern: '^http:\/\/localhost' },
            { pattern: '^https:\/\/YOUR_DATABRICKS_HOST' },
          ]
        : [
            { pattern: '^(?!https?:\/\/(www\.)?mlflow.org|https:\/\/(www\.)?github\.com\/mlflow\/mlflow)' }, // internal links or mlflow/mlflow repo
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
          }
        });

        resolve(brokenLinks);
      }
    });
  });
}
