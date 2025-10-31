import React from 'react';
import Link from '@docusaurus/Link';
import clsx from 'clsx';
import styles from './GlossyCard.module.css';
export var GlossyCard = function (_a) {
    var title = _a.title, description = _a.description, colorTheme = _a.colorTheme, linkPath = _a.linkPath, _b = _a.buttonText, buttonText = _b === void 0 ? 'View documentation' : _b, className = _a.className, icon = _a.icon;
    var colorClass = colorTheme === 'blue' ? 'blueTheme' : 'redTheme';
    return (<div className={clsx(styles.glossyCard, className)}>
      <div className={styles.glossyCardContent}>
        <div className={styles.cardHeader}>
          <div className={clsx(styles.colorBlock, styles[colorClass])}></div>
          {icon && <div className={styles.cardIcon}>{icon}</div>}
          <h2 className={styles.cardTitle}>{title}</h2>
        </div>

        <p className={styles.cardDescription}>{description}</p>

        <div className={styles.cardAction}>
          <Link to={linkPath} className={clsx(styles.cardButton, styles["".concat(colorClass, "Button")])}>
            {buttonText} <span className={styles.arrowIcon}>→</span>
          </Link>
        </div>
      </div>
    </div>);
};
export default GlossyCard;
