import React from 'react';
import { LucideIcon } from 'lucide-react';
import styles from './styles.module.css';

export interface FeatureItem {
  icon?: LucideIcon;
  title: string;
  description: string;
}

interface FeatureHighlightsProps {
  features: FeatureItem[];
}

export default function FeatureHighlights({ features }: FeatureHighlightsProps) {
  return (
    <div className={styles.featureHighlights}>
      {features.map((feature, index) => (
        <div key={index} className={styles.highlightItem}>
          {feature.icon && (
            <div className={styles.highlightIcon}>
              <feature.icon size={24} />
            </div>
          )}
          <div className={styles.highlightContent}>
            <h4>{feature.title}</h4>
            <p>{feature.description}</p>
          </div>
        </div>
      ))}
    </div>
  );
}
