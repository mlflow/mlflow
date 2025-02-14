import { css } from '@emotion/react';
import type { CSSProperties } from 'react';

import { getOffsets, genSkeletonAnimatedColor } from './utils';
import { useDesignSystemTheme } from '../Hooks';
import type { WithLoadingState } from '../LoadingState/LoadingState';
import { LoadingState } from '../LoadingState/LoadingState';
import { visuallyHidden } from '../utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';

interface ParagraphSkeletonProps extends WithLoadingState {
  /** Label for screen readers */
  label?: React.ReactNode;
  /** Seed that deterministically arranges the uneven lines, so that they look like ragged text.
   * If you don't provide this (or give each skeleton the same seed) they will all look the same. */
  seed?: string;
  /** fps for animation. Default is 60 fps. A lower number will use less resources. */
  frameRate?: number;
  /** Style property */
  style?: CSSProperties;
  /** Class name property */
  className?: string;
}

const paragraphContainerStyles = css({
  cursor: 'progress',
  width: '100%',
  height: 20,
  display: 'flex',
  justifyContent: 'flex-start',
  alignItems: 'center',
});

const paragraphFillStyles = css({
  borderRadius: 'var(--border-radius)',
  height: 8,
});

export const ParagraphSkeleton = ({
  label,
  seed = '',
  frameRate = 60,
  style,
  loading = true,
  loadingDescription = 'ParagraphSkeleton',
  ...restProps
}: ParagraphSkeletonProps) => {
  const { theme } = useDesignSystemTheme();
  const offsetWidth = getOffsets(seed)[0];

  return (
    <div
      {...addDebugOutlineIfEnabled()}
      css={paragraphContainerStyles}
      style={{
        ...style,
        ['--border-radius' as any]: `${theme.general.borderRadiusBase}px`,
      }}
      {...restProps}
    >
      {loading && <LoadingState description={loadingDescription} />}
      <span css={visuallyHidden}>{label}</span>
      <div
        aria-hidden
        css={[
          paragraphFillStyles,
          genSkeletonAnimatedColor(theme, frameRate),
          { width: `calc(100% - ${offsetWidth}px)` },
        ]}
      />
    </div>
  );
};
