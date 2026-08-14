import React from 'react';
import { LucideIcon } from 'lucide-react';
import styles from './styles.module.css';

export interface ConceptItem {
  icon?: LucideIcon;
  title: string;
  description: string;
}

interface ConceptOverviewProps {
  concepts: ConceptItem[];
  title?: string;
}

export default function ConceptOverview({ concepts, title }: ConceptOverviewProps) {
  return (
    <div className={styles.conceptOverview}>
      {title && <h3 className={styles.overviewTitle}>{title}</h3>}
      <div className={styles.conceptGrid}>
        {concepts.map((concept, index) => (
          <div key={index} className={styles.conceptCard}>
            <div className={styles.conceptHeader}>
              {concept.icon && (
                <div className={styles.conceptIcon}>
                  <concept.icon size={20} />
                </div>
              )}
              <h4 className={styles.conceptTitle}>{concept.title}</h4>
            </div>
            <p className={styles.conceptDescription}>{concept.description}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
