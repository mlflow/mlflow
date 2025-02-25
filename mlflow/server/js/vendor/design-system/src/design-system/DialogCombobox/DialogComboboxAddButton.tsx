import { css } from '@emotion/react';

import { useDialogComboboxContext } from './hooks/useDialogComboboxContext';
import { Button } from '../Button';
import { useDesignSystemTheme } from '../Hooks';
import { PlusIcon } from '../Icon';
import { getComboboxOptionItemWrapperStyles } from '../_shared_/Combobox';
import { importantify } from '../utils/css-utils';

export const DialogComboboxAddButton = ({ children, ...restProps }: React.HTMLAttributes<HTMLButtonElement>) => {
  const { theme } = useDesignSystemTheme();
  const { isInsideDialogCombobox, componentId } = useDialogComboboxContext();

  if (!isInsideDialogCombobox) {
    throw new Error('`DialogComboboxAddButton` must be used within `DialogCombobox`');
  }

  return (
    <Button
      componentId={`${componentId ? componentId : 'design_system.dialogcombobox'}.add_option`}
      {...restProps}
      type="tertiary"
      className="combobox-footer-add-button"
      css={{
        ...getComboboxOptionItemWrapperStyles(theme),
        ...css(
          importantify({
            width: '100%',
            padding: 0,
            display: 'flex',
            alignItems: 'center',
            borderRadius: 0,

            '&:focus': {
              background: theme.colors.actionTertiaryBackgroundHover,
              outline: 'none',
            },
          }),
        ),
      }}
      icon={<PlusIcon />}
    >
      {children}
    </Button>
  );
};
