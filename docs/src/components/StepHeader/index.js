import React from 'react';
import styles from './styles.module.css';
const StepHeader = ({ number, title }) => {
    return (<div className={styles.stepHeader}>
      <div className={styles.stepNumber}>{number}</div>
      <h3 className={styles.stepTitle}>{title}</h3>
    </div>);
};
export default StepHeader;
