import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { uniqueId } from 'lodash';
import { useEffect, useMemo } from 'react';
import { Text } from './Text';
const MiddleElideSuffixLength = 6;
export function TextMiddleElide({ text, suffixLength = MiddleElideSuffixLength, ...textProps }) {
    const id = useMemo(() => uniqueId('text-middle-elided-'), []);
    const { start, suffix } = getStartAndSuffix(text, suffixLength);
    const disableElide = process?.env?.NODE_ENV === 'test'; // so unit tests play nice
    // use the entire text on select and copy
    useEffect(() => {
        const clipboardCopyHandler = (e) => {
            e?.preventDefault();
            e?.clipboardData?.setData('text/plain', text);
        };
        const selector = `.${id}`;
        document.querySelector(selector)?.addEventListener('copy', clipboardCopyHandler);
        return () => {
            document.querySelector(selector)?.removeEventListener('copy', clipboardCopyHandler);
        };
    }, [id, text]);
    return (_jsx(Text, { ellipsis: disableElide ? undefined : { suffix }, ...textProps, "aria-label": text, title: textProps.title ?? text, className: id, children: disableElide ? text : start }));
}
// Exported for unit tests
export function getStartAndSuffix(text, suffixLength) {
    if (text.length <= suffixLength) {
        return { start: text, suffix: undefined };
    }
    const start = text.slice(0, text.length - suffixLength).trim();
    const suffix = text.slice(-suffixLength).trim();
    return { start, suffix };
}
//# sourceMappingURL=TextMiddleElide.js.map