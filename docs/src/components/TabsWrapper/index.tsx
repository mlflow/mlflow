import React from 'react';
import styles from './TabsWrapper.module.css';

interface TabsWrapperProps {
  children: React.ReactNode;
}

export default function TabsWrapper({ children }: TabsWrapperProps) {
  return <div className={styles.wrapper}>{children}</div>;
}
