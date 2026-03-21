import React from 'react';
import { UNSAFE_NavigationContext } from '../utils/RoutingUtils';

const useNavigationBlock = () => {
  return (React.useContext(UNSAFE_NavigationContext) as any).navigator.block;
};

export interface PromptProps {
  when: boolean;
  message: string;
}

/**
 * Component confirms navigating away by displaying prompt if given condition is met.
 * Uses react-router v6 API.
 */
export const Prompt = ({ when, message }: PromptProps) => {
  const block = useNavigationBlock();

  React.useEffect(() => {
    if (!when) return;

    const unblock = block?.(() => {
      // eslint-disable-next-line no-alert
      return window.confirm(message);
    });

    // eslint-disable-next-line consistent-return
    return unblock;
  }, [message, block, when]);

  return null;
};
