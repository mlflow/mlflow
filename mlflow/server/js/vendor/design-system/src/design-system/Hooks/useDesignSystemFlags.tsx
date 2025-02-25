import { useDesignSystemContext } from './useDesignSystemContext';
import type { DesignSystemFlags } from '../../flags';

export function useDesignSystemFlags(): DesignSystemFlags {
  const context = useDesignSystemContext();

  return context.flags;
}
