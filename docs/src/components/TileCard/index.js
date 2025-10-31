import React from 'react';
import Link from '@docusaurus/Link';
import clsx from 'clsx';
import styles from './styles.module.css';
import useBaseUrl from '@docusaurus/useBaseUrl';
import ThemedImage from '@theme/ThemedImage';
/**
 * A reusable tile card component for displaying feature cards with icon, title, description and link
 */
export default function TileCard(_a) {
    var Icon = _a.icon, image = _a.image, imageDark = _a.imageDark, imageWidth = _a.imageWidth, imageHeight = _a.imageHeight, _b = _a.iconSize, iconSize = _b === void 0 ? 32 : _b, containerHeight = _a.containerHeight, title = _a.title, description = _a.description, href = _a.href, _c = _a.linkText, linkText = _c === void 0 ? 'Learn more →' : _c, className = _a.className;
    // Ensure either icon or image is provided
    if (!Icon && !image) {
        throw new Error('TileCard requires either an icon or image prop');
    }
    var containerStyle = containerHeight ? { height: "".concat(containerHeight, "px") } : {};
    var imageStyle = {};
    if (imageWidth)
        imageStyle.width = "".concat(imageWidth, "px");
    if (imageHeight)
        imageStyle.height = "".concat(imageHeight, "px");
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
