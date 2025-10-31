import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';
export default function TilesGrid(_a) {
    var children = _a.children, className = _a.className;
    return <div className={clsx(styles.tilesGrid, className)}>{children}</div>;
}
