import React, { useState } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import styles from './styles.module.css';
var CollapsibleSection = function (_a) {
    var children = _a.children, title = _a.title, _b = _a.defaultExpanded, defaultExpanded = _b === void 0 ? false : _b;
    var _c = useState(defaultExpanded), isExpanded = _c[0], setIsExpanded = _c[1];
    return (<div className={styles.collapsibleContainer}>
      <div className={styles.header}>
        <button className={styles.toggleButton} onClick={function () { return setIsExpanded(!isExpanded); }} aria-expanded={isExpanded}>
          <span className={styles.toggleText}>{title}</span>
          {isExpanded ? <ChevronUp size={20}/> : <ChevronDown size={20}/>}
        </button>
      </div>
      {isExpanded && <div className={styles.content}>{children}</div>}
    </div>);
};
export default CollapsibleSection;
