import React, { useEffect, useRef, useState } from 'react';

export interface TypographTruncateProps extends React.HTMLAttributes<HTMLSpanElement> {
  children?: React.ReactChild;
  lines?: number;
}

export const Truncate = ({ children, lines = 1, ...props }: TypographTruncateProps) => {
  const [textContent, setTextContent] = useState('');
  const spanRef = useRef<HTMLSpanElement>(null);

  // This ensures that truncated text is always available to the user via a native tooltip
  useEffect(() => {
    if (spanRef.current) {
      setTextContent(spanRef.current.textContent || '');
    }
  }, [spanRef, children]);

  return (
    <span
      ref={spanRef}
      title={textContent}
      css={{
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        whiteSpace: 'normal',
        wordBreak: 'break-word',
        display: '-webkit-box',
        webkitLineClamp: lines,
        WebkitBoxOrient: 'vertical',
        WebkitLineClamp: lines,
      }}
      {...props}
    >
      {children}
    </span>
  );
};
