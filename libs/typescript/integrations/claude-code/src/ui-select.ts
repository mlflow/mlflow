import { emitKeypressEvents } from 'node:readline';

import { bold, cyan, dim } from './ui.js';

export interface SelectOption<T> {
  value: T;
  label: string;
  hint?: string;
}

export interface SelectPromptOptions<T> {
  question: string;
  options: SelectOption<T>[];
  defaultIndex?: number;
  input?: NodeJS.ReadStream;
  output?: NodeJS.WriteStream;
}

export function selectPrompt<T>(opts: SelectPromptOptions<T>): Promise<T> {
  const input = opts.input ?? process.stdin;
  const output = opts.output ?? process.stdout;
  const defaultIndex = opts.defaultIndex ?? 0;
  const { options, question } = opts;

  if (!input.isTTY) {
    return Promise.resolve(options[defaultIndex].value);
  }

  let current = defaultIndex;
  let linesRendered = 0;

  const formatLines = (): string[] => [
    `${bold('?')} ${question}`,
    ...options.map((option, index) => {
      const marker = index === current ? cyan('●') : dim('○');
      const label = index === current ? cyan(option.label) : option.label;
      const hint = option.hint ? `  ${dim(option.hint)}` : '';
      const defaultTag = index === defaultIndex ? dim('  (default)') : '';
      return `  ${marker} ${label}${hint}${defaultTag}`;
    }),
    '',
    `  ${dim('↑/↓ to move, enter to select')}`,
  ];

  const render = (): void => {
    if (linesRendered > 0) {
      output.write(`\x1b[${linesRendered}A\x1b[0J`);
    }
    const lines = formatLines();
    output.write(lines.join('\n') + '\n');
    linesRendered = lines.length;
  };

  emitKeypressEvents(input);
  const wasRaw = input.isRaw;
  input.setRawMode(true);
  input.resume();

  return new Promise<T>((resolvePromise) => {
    const cleanup = (): void => {
      input.removeListener('keypress', onKeypress);
      if (!wasRaw) {
        input.setRawMode(false);
      }
      input.pause();
    };

    const onKeypress = (_str: string, key: { name?: string; ctrl?: boolean }): void => {
      if (key.ctrl && key.name === 'c') {
        cleanup();
        process.exit(130);
      } else if (key.name === 'up' || key.name === 'k') {
        current = (current - 1 + options.length) % options.length;
        render();
      } else if (key.name === 'down' || key.name === 'j') {
        current = (current + 1) % options.length;
        render();
      } else if (key.name === 'return') {
        cleanup();
        resolvePromise(options[current].value);
      }
    };

    input.on('keypress', onKeypress);
    render();
  });
}
