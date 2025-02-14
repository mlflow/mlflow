import type { UseComboboxGetToggleButtonPropsOptions } from 'downshift';

import { TypeaheadComboboxToggleButton, type DownshiftToggleButtonProps } from './TypeaheadComboboxToggleButton';
import { useDesignSystemTheme } from '../Hooks';
import { ClearSelectionButton } from '../_shared_/Combobox/ClearSelectionButton';
export interface TypeaheadComboboxControlsProps {
  getDownshiftToggleButtonProps: (options?: UseComboboxGetToggleButtonPropsOptions) => DownshiftToggleButtonProps;
  showClearSelectionButton?: boolean;
  showComboboxToggleButton?: boolean;
  handleClear?: (e: any) => void;
  disabled?: boolean;
}

export const TypeaheadComboboxControls: React.FC<any> = ({
  getDownshiftToggleButtonProps,
  showClearSelectionButton,
  showComboboxToggleButton = true,
  handleClear,
  disabled,
}: TypeaheadComboboxControlsProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        position: 'absolute',
        top: theme.spacing.sm,
        right: 7,
        height: 16,
        zIndex: 1,
      }}
    >
      {showClearSelectionButton && (
        <ClearSelectionButton
          onClick={handleClear}
          css={{
            pointerEvents: 'all',
            verticalAlign: 'text-top',
          }}
        />
      )}
      {showComboboxToggleButton && (
        <TypeaheadComboboxToggleButton {...getDownshiftToggleButtonProps()} disabled={disabled} />
      )}
    </div>
  );
};
