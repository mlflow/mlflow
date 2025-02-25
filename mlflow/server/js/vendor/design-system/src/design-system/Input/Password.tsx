import { Input as AntDInput } from 'antd';
import React, { forwardRef } from 'react';

import { getInputEmotionStyles } from './Input';
import type { PasswordProps } from './common';
import { DesignSystemAntDConfigProvider } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks/useDesignSystemTheme';
import { useDesignSystemSafexFlags } from '../utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';

export const Password: React.FC<PasswordProps> = forwardRef<AntDInput, PasswordProps>(function Password(
  { validationState, autoComplete = 'off', dangerouslySetAntdProps, dangerouslyAppendEmotionCSS, ...props },
  ref,
) {
  const { classNamePrefix, theme } = useDesignSystemTheme();
  const { useNewShadows } = useDesignSystemSafexFlags();

  return (
    <DesignSystemAntDConfigProvider>
      <AntDInput.Password
        {...addDebugOutlineIfEnabled()}
        visibilityToggle={false}
        ref={ref}
        autoComplete={autoComplete}
        css={[
          getInputEmotionStyles(classNamePrefix, theme, { validationState, useNewShadows }),
          dangerouslyAppendEmotionCSS,
        ]}
        {...props}
        {...dangerouslySetAntdProps}
      />
    </DesignSystemAntDConfigProvider>
  );
});
