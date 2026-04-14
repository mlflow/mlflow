import React from 'react';
import styles from './styles.module.css';
export default function FeatureHighlights({ features, col = 2 }) {
    return (<div className={styles.featureHighlights} style={{ gridTemplateColumns: `repeat(${col}, 1fr)` }}>
      {features.map((feature, index) => (<div key={index} className={styles.highlightItem}>
          {feature.icon && (<div className={styles.highlightIcon}>
              <feature.icon size={24}/>
            </div>)}
          <div className={styles.highlightContent}>
            <h4>{feature.title}</h4>
            <p>{feature.description}</p>
          </div>
        </div>))}
    </div>);
}
