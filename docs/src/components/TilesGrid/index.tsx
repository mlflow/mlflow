import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

export interface TilesGridProps {
  children: React.ReactNode;
  className?: string;
  cols?: number;
}

export default function TilesGrid({ children, className, cols = 3 }: TilesGridProps): JSX.Element {
  const gridTemplateColumns = `repeat(${cols}, 1fr)`;
  return (
    <div className={clsx(styles.tilesGrid, className)} style={{ gridTemplateColumns }}>
      {children}
    </div>
  );
}
