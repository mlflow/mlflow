import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { useEffect, useRef, useState } from 'react';
export const Truncate = ({ children, lines = 1, ...props }) => {
    const [textContent, setTextContent] = useState('');
    const spanRef = useRef(null);
    // This ensures that truncated text is always available to the user via a native tooltip
    useEffect(() => {
        if (spanRef.current) {
            setTextContent(spanRef.current.textContent || '');
        }
    }, [spanRef, children]);
    return (_jsx("span", { ref: spanRef, title: textContent, css: {
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'normal',
            wordBreak: 'break-word',
            display: '-webkit-box',
            webkitLineClamp: lines,
            WebkitBoxOrient: 'vertical',
            WebkitLineClamp: lines,
        }, ...props, children: children }));
};
//# sourceMappingURL=Truncate.js.map