import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

export interface TilesGridProps {
  children: React.ReactNode;
  className?: string;
}

export default function TilesGrid({ children, className }: TilesGridProps): JSX.Element {
  return <div className={clsx(styles.tilesGrid, className)}>{children}</div>;
}
