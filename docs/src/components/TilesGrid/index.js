import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';
export default function TilesGrid({ children, className }) {
    return <div className={clsx(styles.tilesGrid, className)}>{children}</div>;
}
