/**
 * Read all data from stdin and parse as JSON.
 */
export function readStdin<T>(): Promise<T> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    process.stdin.on('data', (chunk: Buffer) => chunks.push(chunk));
    process.stdin.on('end', () => {
      try {
        const raw = Buffer.concat(chunks).toString('utf-8');
        resolve(JSON.parse(raw) as T);
      } catch (err) {
        reject(new Error(`Failed to parse stdin as JSON: ${String(err)}`));
      }
    });
    process.stdin.on('error', reject);
  });
}
