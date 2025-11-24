import { useStable } from './useStable';
let sequentialCounter = 0;
export function useStableUid() {
    return useStable(() => sequentialCounter++);
}
//# sourceMappingURL=useStableUid.js.map