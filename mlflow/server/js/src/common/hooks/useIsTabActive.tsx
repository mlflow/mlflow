import { useEffect, useState } from 'react';

/**
 * Hook that returns whether the browser tab is active or not.
 * @returns true if the tab is active, false otherwise
 */
export const useIsTabActive = () => {
  const [isTabActive, setIsTabActive] = useState(document.visibilityState === 'visible');
  useEffect(() => {
    document.addEventListener('visibilitychange', (x) => {
      setIsTabActive(document.visibilityState === 'visible');
    });
  }, []);
  return isTabActive;
};
