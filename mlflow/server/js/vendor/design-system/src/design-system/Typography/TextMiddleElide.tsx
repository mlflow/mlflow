import { uniqueId } from 'lodash';
import { useEffect, useMemo } from 'react';

import { Text } from './Text';
import type { TypographyTextProps } from './Text';

const MiddleElideSuffixLength = 6;

export type TypographyTextMiddleElideProps = Omit<TypographyTextProps, 'ellipsis' | 'children'> & {
  text: string;
  suffixLength?: number;
};

export function TextMiddleElide({
  text,
  suffixLength = MiddleElideSuffixLength,
  ...textProps
}: TypographyTextMiddleElideProps): JSX.Element {
  const id = useMemo(() => uniqueId('text-middle-elided-'), []);
  const { start, suffix } = getStartAndSuffix(text, suffixLength);
  const disableElide = process?.env?.NODE_ENV === 'test'; // so unit tests play nice

  // use the entire text on select and copy
  useEffect(() => {
    const clipboardCopyHandler = (e: ClipboardEvent) => {
      e?.preventDefault();
      e?.clipboardData?.setData('text/plain', text);
    };

    const selector = `.${id}`;
    document.querySelector(selector)?.addEventListener('copy', clipboardCopyHandler as any);
    return () => {
      document.querySelector(selector)?.removeEventListener('copy', clipboardCopyHandler as any);
    };
  }, [id, text]);

  return (
    <Text
      ellipsis={disableElide ? undefined : { suffix }}
      {...textProps}
      aria-label={text}
      title={textProps.title ?? text}
      className={id}
    >
      {disableElide ? text : start}
    </Text>
  );
}

// Exported for unit tests
export function getStartAndSuffix(text: string, suffixLength: number) {
  if (text.length <= suffixLength) {
    return { start: text, suffix: undefined };
  }
  const start = text.slice(0, text.length - suffixLength).trim();
  const suffix = text.slice(-suffixLength).trim();
  return { start, suffix };
}
