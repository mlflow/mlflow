import React from 'react';
import clsx from 'clsx';
import Buttons from '@theme/CodeBlock/Buttons';
import styles from './styles.module.css';
export default function ContentHeader(_a) {
    var language = _a.language;
    return (<div className={clsx(styles.codeBlockHeader)} aria-label={"Code block header for ".concat(language, " code with copy and toggle buttons")}>
      <span className={styles.languageLabel}>{language}</span>
      <Buttons />
    </div>);
}
