import React from 'react';
import { useHomePageViewState } from '../HomePageViewStateContext';

const LazyLogTracesDrawer = React.lazy(() => import('./LogTracesDrawer'));

/**
 * Loads expensive LogTracesDrawer component only when the drawer is open.
 */
export const LogTracesDrawerLoader = () => {
  const { isLogTracesDrawerOpen } = useHomePageViewState();

  if (isLogTracesDrawerOpen) {
    return (
      <React.Suspense fallback={null}>
        <LazyLogTracesDrawer />
      </React.Suspense>
    );
  }
  return null;
};

export default LogTracesDrawerLoader;
