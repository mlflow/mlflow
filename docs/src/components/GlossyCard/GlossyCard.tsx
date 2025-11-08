import React from 'react';
import Link from '@docusaurus/Link';
import clsx from 'clsx';
import styles from './GlossyCard.module.css';

interface GlossyCardProps {
  /**
   * The title displayed in the card header
   */
  title: string;

  /**
   * Descriptive text for the card content
   */
  description: string;

  /**
   * Color theme for the card ('blue' or 'red')
   */
  colorTheme: 'blue' | 'red';

  /**
   * The URL that the card button will link to
   */
  linkPath: string;

  /**
   * Custom text for the card button (default: "View documentation")
   */
  buttonText?: string;

  /**
   * Additional CSS class names to apply to the card
   */
  className?: string;

  /**
   * Optional icon to display next to the title
   */
  icon?: React.ReactNode;
}

export const GlossyCard: React.FC<GlossyCardProps> = ({
  title,
  description,
  colorTheme,
  linkPath,
  buttonText = 'View documentation',
  className,
  icon,
}) => {
  const colorClass = colorTheme === 'blue' ? 'blueTheme' : 'redTheme';

  return (
    <div className={clsx(styles.glossyCard, className)}>
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
    </div>
  );
};

export default GlossyCard;
