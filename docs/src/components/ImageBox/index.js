import React from 'react';
import styles from './ImageBox.module.css';
import useBaseUrl from '@docusaurus/useBaseUrl';
export default function ImageBox(_a) {
    var src = _a.src, alt = _a.alt, width = _a.width, caption = _a.caption, className = _a.className;
    return (<div className={"".concat(styles.container, " ").concat(className || '')}>
      <div className={styles.imageWrapper} style={width ? { width: width } : {}}>
        <img src={useBaseUrl(src)} alt={alt} className={styles.image}/>
      </div>
      {caption && <p className={styles.caption}>{caption}</p>}
    </div>);
}
