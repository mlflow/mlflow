import React, { useState } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import styles from './styles.module.css';

interface CollapsibleSectionProps {
  children: React.ReactNode;
  title: string;
  defaultExpanded?: boolean;
}

const CollapsibleSection: React.FC<CollapsibleSectionProps> = ({ children, title, defaultExpanded = false }) => {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  return (
    <div className={styles.collapsibleContainer}>
      <div className={styles.header}>
        <button className={styles.toggleButton} onClick={() => setIsExpanded(!isExpanded)} aria-expanded={isExpanded}>
          <span className={styles.toggleText}>{title}</span>
          {isExpanded ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
        </button>
      </div>
      {isExpanded && <div className={styles.content}>{children}</div>}
    </div>
  );
};

export default CollapsibleSection;
