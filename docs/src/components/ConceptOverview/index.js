import React from 'react';
import styles from './styles.module.css';
export default function ConceptOverview({ concepts, title }) {
    return (<div className={styles.conceptOverview}>
      {title && <h3 className={styles.overviewTitle}>{title}</h3>}
      <div className={styles.conceptGrid}>
        {concepts.map((concept, index) => (<div key={index} className={styles.conceptCard}>
            <div className={styles.conceptHeader}>
              {concept.icon && (<div className={styles.conceptIcon}>
                  <concept.icon size={20}/>
                </div>)}
              <h4 className={styles.conceptTitle}>{concept.title}</h4>
            </div>
            <p className={styles.conceptDescription}>{concept.description}</p>
          </div>))}
      </div>
    </div>);
}
