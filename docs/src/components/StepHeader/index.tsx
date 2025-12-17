import React from 'react';
import styles from './styles.module.css';

interface StepHeaderProps {
  number: number;
  title: string;
}

const StepHeader: React.FC<StepHeaderProps> = ({ number, title }) => {
  return (
    <div className={styles.stepHeader}>
      <div className={styles.stepNumber}>{number}</div>
      <h3 className={styles.stepTitle}>{title}</h3>
    </div>
  );
};

export default StepHeader;
