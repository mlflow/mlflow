import type { SkeletonProps as AntDSkeletonProps } from 'antd';
import { Skeleton as AntDSkeleton } from 'antd';

import { AccessibleContainer } from '../AccessibleContainer';
import { DesignSystemAntDConfigProvider } from '../DesignSystemProvider';
import type { WithLoadingState } from '../LoadingState/LoadingState';
import { LoadingState } from '../LoadingState/LoadingState';
import type { DangerouslySetAntdProps, HTMLDataAttributes } from '../types';

export interface LegacySkeletonProps
  extends WithLoadingState,
    AntDSkeletonProps,
    DangerouslySetAntdProps<AntDSkeletonProps>,
    HTMLDataAttributes {
  label?: React.ReactNode;
}

interface LegacySkeletonInterface extends React.FC<LegacySkeletonProps> {
  Button: typeof AntDSkeleton.Button;
  Image: typeof AntDSkeleton.Image;
  Input: typeof AntDSkeleton.Input;
}

/** @deprecated This component is deprecated. Use ParagraphSkeleton, TitleSkeleton, or GenericSkeleton instead. */
export const LegacySkeleton = /* #__PURE__ */ (() => {
  const LegacySkeleton: LegacySkeletonInterface = ({
    dangerouslySetAntdProps,
    label,
    loadingDescription = 'LegacySkeleton',
    ...props
  }) => {
    // There is a conflict in how the 'loading' prop is handled here, so we can't default it to true in
    // props destructuring above like we do for 'loadingDescription'. The 'loading' param is used both
    // for <LoadingState> and in <AntDSkeleton>. The intent is for 'loading' to default to true in
    // <LoadingState>, but if we do that, <AntDSkeleton> will not render the children. The workaround
    // here is to default 'loading' to true only when considering whether to render a <LoadingState>.
    // Also, AntDSkeleton looks at the presence of 'loading' in props, so We cannot explicitly destructure
    // 'loading' in the constructor since we would no longer be able to differentiate between it not being
    // passed in at all and being passed undefined.
    const loadingStateLoading = (props as LegacySkeletonProps).loading ?? true;

    return (
      <DesignSystemAntDConfigProvider>
        <AccessibleContainer label={label}>
          {loadingStateLoading && <LoadingState description={loadingDescription} />}
          <AntDSkeleton {...props} {...dangerouslySetAntdProps} />
        </AccessibleContainer>
      </DesignSystemAntDConfigProvider>
    );
  };

  LegacySkeleton.Button = AntDSkeleton.Button;
  LegacySkeleton.Image = AntDSkeleton.Image;
  LegacySkeleton.Input = AntDSkeleton.Input;

  return LegacySkeleton;
})();
