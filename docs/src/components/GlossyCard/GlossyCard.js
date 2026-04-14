import React from 'react';
import Link from '@docusaurus/Link';
import clsx from 'clsx';
import styles from './GlossyCard.module.css';
export const GlossyCard = ({ title, description, colorTheme, linkPath, buttonText = 'View documentation', className, icon, }) => {
    const colorClass = colorTheme === 'blue' ? 'blueTheme' : 'redTheme';
    return (<div className={clsx(styles.glossyCard, className)}>
      <div className={styles.glossyCardContent}>
        <div className={styles.cardHeader}>
          <div className={clsx(styles.colorBlock, styles[colorClass])}></div>
          {icon && <div className={styles.cardIcon}>{icon}</div>}
          <h2 className={styles.cardTitle}>{title}</h2>
        </div>

        <p className={styles.cardDescription}>{description}</p>

        <div className={styles.cardAction}>
          <Link to={linkPath} className={clsx(styles.cardButton, styles[`${colorClass}Button`])}>
            {buttonText} <span className={styles.arrowIcon}>→</span>
          </Link>
        </div>
      </div>
    </div>);
};
export default GlossyCard;
