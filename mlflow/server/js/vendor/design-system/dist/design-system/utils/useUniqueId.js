import { useState } from 'react';
const uniqueId = () => new Date().getTime() +
    Array(16)
        .fill('')
        .map(() => parseInt((Math.random() * 10).toString()))
        .join('');
export function useUniqueId(prefix = '') {
    return useState(() => `${prefix}-${uniqueId()}`)[0];
}
//# sourceMappingURL=useUniqueId.js.map