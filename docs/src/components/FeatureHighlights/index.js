import React from 'react';
import styles from './styles.module.css';
export default function FeatureHighlights(_a) {
    var features = _a.features;
    return (<div className={styles.featureHighlights}>
      {features.map(function (feature, index) { return (<div key={index} className={styles.highlightItem}>
          {feature.icon && (<div className={styles.highlightIcon}>
              <feature.icon size={24}/>
            </div>)}
          <div className={styles.highlightContent}>
            <h4>{feature.title}</h4>
            <p>{feature.description}</p>
          </div>
        </div>); })}
    </div>);
}
