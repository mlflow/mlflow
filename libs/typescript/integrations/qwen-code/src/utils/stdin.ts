/**
 * Read JSON from stdin (hook input).
 * Qwen Code hooks receive JSON payloads via stdin.
 */
export function readStdin<T>(): Promise<T> {
  return new Promise((resolve, reject) => {
    let data = '';
    process.stdin.setEncoding('utf-8');
    process.stdin.on('data', (chunk: string) => {
      data += chunk;
    });
    process.stdin.on('end', () => {
      try {
        resolve(JSON.parse(data) as T);
      } catch (err) {
        reject(new Error(`Failed to parse stdin JSON: ${String(err)}`));
      }
    });
    process.stdin.on('error', reject);
  });
}
