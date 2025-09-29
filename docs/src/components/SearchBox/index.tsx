import React, { useState, useEffect } from 'react';
import { Search } from 'lucide-react';
import clsx from 'clsx';
import styles from './styles.module.css';

export interface SearchBoxProps {
  /**
   * Placeholder text for the button
   */
  placeholder?: string;
  /**
   * Additional CSS classes
   */
  className?: string;
}

export default function SearchBox({
  placeholder = 'What do you want to learn?',
  className,
}: SearchBoxProps): JSX.Element {
  const openRunLLM = () => {
    // Open RunLLM widget when user clicks the input
    if (typeof window !== 'undefined' && (window as any).runllm) {
      (window as any).runllm.open();
    }
  };

  return (
    <div className={clsx(styles.searchContainer, className)}>
      <div className={styles.searchWrapper}>
        <input type="text" className={styles.searchInput} placeholder={placeholder} onClick={openRunLLM} readOnly />
        <button type="button" className={styles.searchButton} onClick={openRunLLM}>
          <Search size={20} />
        </button>
      </div>
    </div>
  );
}
