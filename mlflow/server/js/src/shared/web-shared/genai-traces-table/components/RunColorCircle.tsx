import React from 'react';

export const RunColorCircle = React.memo(({ color, hidden }: { color: string; hidden?: boolean }) => {
  return (
    <label
      css={{
        width: 12,
        height: 12,
        borderRadius: 6,
        flexShrink: 0,
        border: '1px solid rgba(0, 0, 0, 0.1)',
        cursor: 'default',
        position: 'relative',
      }}
      style={{
        backgroundColor: color,
        display: hidden ? 'none' : '',
      }}
    >
      <span
        css={{
          clip: 'rect(0px, 0px, 0px, 0px)',
          clipPath: 'inset(50%)',
          height: '1px',
          overflow: 'hidden',
          position: 'absolute',
          whiteSpace: 'nowrap',
          width: '1px',
        }}
      >
        {color}
      </span>
    </label>
  );
});
