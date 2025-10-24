import React, { type ReactNode } from 'react';
import clsx from 'clsx';
import Buttons from '@theme/CodeBlock/Buttons';
import styles from './styles.module.css';

export interface ContentHeaderProps {
  language?: string;
}

export default function ContentHeader({ language }: ContentHeaderProps): ReactNode {
  return (
    <div className={clsx(styles.codeBlockHeader)}>
      <span className={styles.languageLabel}>{language}</span>
      <Buttons />
    </div>
  );
}
