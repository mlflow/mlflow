import React from 'react';
import clsx from 'clsx';
import { useCodeBlockContext } from '@docusaurus/theme-common/internal';
import Container from '@theme/CodeBlock/Container';
import Title from '@theme/CodeBlock/Title';
import Content from '@theme/CodeBlock/Content';
import ContentHeader from '../ContentHeader';
import styles from './styles.module.css';
export default function CodeBlockLayout({ className }) {
    const { metadata } = useCodeBlockContext();
    const language = metadata.language || 'text'; // Use 'text' as a fallback when metadata.language is undefined.
    return (<Container as="div" className={clsx(className, metadata.className)}>
      {metadata.title && (<div className={styles.codeBlockTitle}>
          <Title>{metadata.title}</Title>
        </div>)}
      <div className={styles.codeBlockContent}>
        <ContentHeader language={language}/>
        <Content />
      </div>
    </Container>);
}
