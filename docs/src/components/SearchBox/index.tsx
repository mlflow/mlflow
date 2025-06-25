import React, { useState, useEffect } from 'react';
import { Search } from 'lucide-react';
import clsx from 'clsx';
import styles from './styles.module.css';

export interface SearchBoxProps {
  /**
   * RunLLM Assistant ID
   */
  assistantId: string;
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
  assistantId,
  placeholder = "What do you want to learn?",
  className,
}: SearchBoxProps): JSX.Element {
  const [query, setQuery] = useState('');

  // Initialize RunLLM widget
  useEffect(() => {
    // Check if script already exists
    if (document.getElementById('runllm-widget-script')) {
      return;
    }

    // Add RunLLM widget script
    const script = document.createElement('script');
    script.type = 'module';
    script.id = 'runllm-widget-script';
    script.src = 'https://widget.runllm.com';
    script.setAttribute('runllm-assistant-id', assistantId);
    script.setAttribute('runllm-position', 'BOTTOM_RIGHT');
    script.async = true;
    document.head.appendChild(script);

    return () => {
      // Cleanup if needed
      const existingScript = document.getElementById('runllm-widget-script');
      if (existingScript && existingScript.parentNode) {
        existingScript.parentNode.removeChild(existingScript);
      }
    };
  }, [assistantId]);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    
    // Open RunLLM widget with the query if provided
    if (typeof window !== 'undefined' && (window as any).runllm) {
      (window as any).runllm.open();
      
      // If there's a query, we could potentially pass it to RunLLM
      // This depends on RunLLM's API - for now we just open the widget
      if (query.trim()) {
        // Clear the input after opening the widget
        setQuery('');
      }
    }
  };

  const handleInputClick = () => {
    // Open RunLLM widget when user clicks the input
    if (typeof window !== 'undefined' && (window as any).runllm) {
      (window as any).runllm.open();
    }
  };

  return (
    <div className={clsx(styles.searchContainer, className)}>
      <form onSubmit={handleSearch} className={styles.searchForm}>
        <input
          type="text"
          className={styles.searchInput}
          placeholder={placeholder}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onClick={handleInputClick}
          readOnly
        />
        <button type="submit" className={styles.searchButton}>
          <Search size={20} />
        </button>
      </form>
    </div>
  );
}