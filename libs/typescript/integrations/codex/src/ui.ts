/**
 * Tiny ANSI styling helpers for the `mlflow-codex` CLI.
 *
 * No runtime dependency on `chalk`/`kleur`: we keep the bundle small and
 * respect the usual opt-outs (non-TTY stdout, `NO_COLOR`, `FORCE_COLOR=0`).
 */

const COLOR_ENABLED = (() => {
  if (process.env.FORCE_COLOR === '0' || process.env.NO_COLOR) {
    return false;
  }
  if (process.env.FORCE_COLOR) {
    return true;
  }
  return Boolean(process.stdout.isTTY);
})();

function wrap(code: string): (s: string) => string {
  return (s: string) => (COLOR_ENABLED ? `\x1b[${code}m${s}\x1b[0m` : s);
}

export const bold = wrap('1');
export const dim = wrap('2');
export const green = wrap('32');
export const red = wrap('31');
export const cyan = wrap('36');
export const yellow = wrap('33');

export const OK = green('✓');
export const FAIL = red('✗');
export const WARN = yellow('!');
