import { useContext } from 'react';

import type { DuboisContextType } from '../DesignSystemProvider';
import { DesignSystemContext } from '../DesignSystemProvider';

export function useDesignSystemContext(): DuboisContextType {
  return useContext(DesignSystemContext);
}
