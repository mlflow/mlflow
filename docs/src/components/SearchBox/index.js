import React from 'react';
import { Search } from 'lucide-react';
import clsx from 'clsx';
import styles from './styles.module.css';
export default function SearchBox({ placeholder = 'What do you want to learn?', className, }) {
    const openRunLLM = () => {
        // Open RunLLM widget when user clicks the input
        if (typeof window !== 'undefined' && window.runllm) {
            window.runllm.open();
        }
    };
    return (<div className={clsx(styles.searchContainer, className)}>
      <div className={styles.searchWrapper}>
        <input type="text" className={styles.searchInput} placeholder={placeholder} onClick={openRunLLM} readOnly/>
        <button type="button" className={styles.searchButton} onClick={openRunLLM}>
          <Search size={20}/>
        </button>
      </div>
    </div>);
}
