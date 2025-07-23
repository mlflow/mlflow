import React, { useState } from 'react';
import { LucideIcon, ChevronDown, ChevronUp } from 'lucide-react';
import styles from './styles.module.css';
import ImageBox from '../ImageBox';

interface WorkflowStep {
  title: string;
  description: string;
  icon?: LucideIcon;
}

interface WorkflowStepsProps {
  steps: WorkflowStep[];
  title?: string;
  screenshot?: {
    src: string;
    alt: string;
    width?: string;
  };
  defaultExpanded?: boolean;
}

const WorkflowSteps: React.FC<WorkflowStepsProps> = ({ steps, title, screenshot, defaultExpanded = false }) => {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  return (
    <div className={styles.workflowContainer}>
      {title && <h3 className={styles.workflowTitle}>{title}</h3>}
      {screenshot && (
        <div className={styles.screenshotContainer}>
          <ImageBox src={screenshot.src} alt={screenshot.alt} width={screenshot.width || '90%'} />
        </div>
      )}
      <div className={styles.stepsHeader}>
        <button className={styles.toggleButton} onClick={() => setIsExpanded(!isExpanded)} aria-expanded={isExpanded}>
          <span className={styles.toggleText}>
            {isExpanded ? 'Hide' : 'Show'} Step-by-Step Instructions ({steps.length} steps)
          </span>
          {isExpanded ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
        </button>
      </div>
      {isExpanded && (
        <div className={styles.stepsContainer}>
          {steps.map((step, index) => (
            <div key={index} className={styles.stepItem}>
              <div className={styles.stepIndicator}>
                <div className={styles.stepNumber}>
                  {step.icon ? <step.icon size={16} /> : <span className={styles.stepNumberText}>{index + 1}</span>}
                </div>
                {index < steps.length - 1 && <div className={styles.stepConnector} />}
              </div>
              <div className={styles.stepContent}>
                <h4 className={styles.stepTitle}>{step.title}</h4>
                <p className={styles.stepDescription}>{step.description}</p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default WorkflowSteps;
