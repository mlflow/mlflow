import React from 'react';
import styles from './styles.module.css';
import ImageBox from '../ImageBox';
var WorkflowSteps = function (_a) {
    var steps = _a.steps, title = _a.title, screenshot = _a.screenshot, _b = _a.width, width = _b === void 0 ? 'normal' : _b;
    return (<div className={styles.workflowContainer}>
      {title && <h3 className={styles.workflowTitle}>{title}</h3>}
      {screenshot && (<div className={styles.screenshotContainer}>
          <ImageBox src={screenshot.src} alt={screenshot.alt} width={screenshot.width || '90%'}/>
        </div>)}
      <div className={styles.stepsContainer} style={{ maxWidth: width === 'wide' ? '850px' : '700px' }}>
        {steps.map(function (step, index) { return (<div key={index} className={styles.stepItem}>
            <div className={styles.stepIndicator}>
              <div className={styles.stepNumber}>
                {step.icon ? <step.icon size={16}/> : <span className={styles.stepNumberText}>{index + 1}</span>}
              </div>
              {index < steps.length - 1 && <div className={styles.stepConnector}/>}
            </div>
            <div className={styles.stepContent}>
              <h4 className={styles.stepTitle}>{step.title}</h4>
              <p className={styles.stepDescription}>{step.description}</p>
            </div>
          </div>); })}
      </div>
    </div>);
};
export default WorkflowSteps;
