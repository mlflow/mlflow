import React from 'react';
import Link from '@docusaurus/Link';
import clsx from 'clsx';
import styles from './styles.module.css';
import useBaseUrl from '@docusaurus/useBaseUrl';
import ThemedImage from '@theme/ThemedImage';
/**
 * A reusable tile card component for displaying feature cards with icon, title, description and link
 */
export default function TileCard({ icon: Icon, image, imageDark, imageWidth, imageHeight, iconSize = 32, containerHeight, title, description, href, linkText = 'Learn more →', className, }) {
    // Ensure either icon or image is provided
    if (!Icon && !image) {
        throw new Error('TileCard requires either an icon or image prop');
    }
    const containerStyle = containerHeight ? { height: `${containerHeight}px` } : {};
    const imageStyle = {};
    if (imageWidth)
        imageStyle.width = `${imageWidth}px`;
    if (imageHeight)
        imageStyle.height = `${imageHeight}px`;
    return (<Link href={href} className={clsx(styles.tileCard, className)}>
      <div className={styles.tileIcon} style={containerStyle}>
        {Icon ? (<Icon size={iconSize}/>) : imageDark ? (<ThemedImage sources={{
                light: useBaseUrl(image),
                dark: useBaseUrl(imageDark),
            }} alt={title} className={styles.tileImage} style={imageStyle}/>) : (<img src={useBaseUrl(image)} alt={title} className={styles.tileImage} style={imageStyle}/>)}
      </div>
      <h3>{title}</h3>
      <p>{description}</p>
      <div className={styles.tileLink}>{linkText}</div>
    </Link>);
}
