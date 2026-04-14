import React from 'react';
import styles from './TabsWrapper.module.css';
export default function TabsWrapper({ children }) {
    return <div className={styles.wrapper}>{children}</div>;
}
