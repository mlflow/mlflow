import { css } from '@emotion/react';
import type { CSSProperties } from 'react';

import { genSkeletonAnimatedColor } from './utils';
import { useDesignSystemTheme } from '../Hooks';
import type { WithLoadingState } from '../LoadingState/LoadingState';
import { LoadingState } from '../LoadingState/LoadingState';
import { visuallyHidden } from '../utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';

interface GenericSkeletonProps extends WithLoadingState {
  /** Label for screen readers */
  label?: React.ReactNode;
  /** fps for animation. Default is 60 fps. A lower number will use less resources. */
  frameRate?: number;
  /** Style property */
  style?: CSSProperties;
  /** Class name property */
  className?: string;
}

const GenericContainerStyles = css({
  cursor: 'progress',
  borderRadius: 'var(--border-radius)',
});

export const GenericSkeleton = ({
  label,
  frameRate = 60,
  style,
  loading = true,
  loadingDescription = 'GenericSkeleton',
  ...restProps
}: GenericSkeletonProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      {...addDebugOutlineIfEnabled()}
      css={[GenericContainerStyles, genSkeletonAnimatedColor(theme, frameRate)]}
      style={{
        ...style,
        ['--border-radius' as any]: `${theme.general.borderRadiusBase}px`,
      }}
      {...restProps}
    >
      {loading && <LoadingState description={loadingDescription} />}
      <span css={visuallyHidden}>{label}</span>
    </div>
  );
};
