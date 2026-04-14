import React from 'react';
import styles from './ImageBox.module.css';
import useBaseUrl from '@docusaurus/useBaseUrl';
export default function ImageBox({ src, alt, width, caption, className }) {
    return (<div className={`${styles.container} ${className || ''}`}>
      <div className={styles.imageWrapper} style={width ? { width } : {}}>
        <img src={useBaseUrl(src)} alt={alt} className={styles.image}/>
      </div>
      {caption && <p className={styles.caption}>{caption}</p>}
    </div>);
}
