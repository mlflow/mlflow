import { css } from '@emotion/react';
import type { CSSProperties } from 'react';

import { genSkeletonAnimatedColor } from './utils';
import { useDesignSystemTheme } from '../Hooks';
import { LoadingState } from '../LoadingState/LoadingState';
import type { WithLoadingState } from '../LoadingState/LoadingState';
import { visuallyHidden } from '../utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';

interface TitleSkeletonProps extends WithLoadingState {
  /** Label for screen readers */
  label?: React.ReactNode;
  /** fps for animation. Default is 60 fps. A lower number will use less resources. */
  frameRate?: number;
  /** Style property */
  style?: CSSProperties;
  /** Class name property */
  className?: string;
}

const titleContainerStyles = css({
  cursor: 'progress',
  width: '100%',
  height: 28,
  display: 'flex',
  justifyContent: 'flex-start',
  alignItems: 'center',
});

const titleFillStyles = css({
  borderRadius: 'var(--border-radius)',
  height: 12,
  width: '100%',
});

export const TitleSkeleton = ({
  label,
  frameRate = 60,
  style,
  loading = true,
  loadingDescription = 'TitleSkeleton',
  ...restProps
}: TitleSkeletonProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      {...addDebugOutlineIfEnabled()}
      css={titleContainerStyles}
      style={{
        ...style,
        ['--border-radius' as any]: `${theme.general.borderRadiusBase}px`,
      }}
      {...restProps}
    >
      {loading && <LoadingState description={loadingDescription} />}
      <span css={visuallyHidden}>{label}</span>
      <div aria-hidden css={[titleFillStyles, genSkeletonAnimatedColor(theme, frameRate)]} />
    </div>
  );
};
