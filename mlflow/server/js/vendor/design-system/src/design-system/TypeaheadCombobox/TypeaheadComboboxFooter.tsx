import { useTypeaheadComboboxContext } from './hooks';
import { useDesignSystemTheme } from '../Hooks';
import { getFooterStyles } from '../_shared_/Combobox';

export interface TypeaheadComboboxFooterProps extends React.HTMLAttributes<HTMLDivElement> {
  _TYPE?: string;
}

export const DuboisTypeaheadComboboxFooter = ({ children, ...restProps }: TypeaheadComboboxFooterProps) => {
  const { theme } = useDesignSystemTheme();
  const { isInsideTypeaheadCombobox } = useTypeaheadComboboxContext();

  if (!isInsideTypeaheadCombobox) {
    throw new Error('`TypeaheadComboboxFooter` must be used within `TypeaheadComboboxMenu`');
  }

  return (
    <div {...restProps} css={getFooterStyles(theme)}>
      {children}
    </div>
  );
};

DuboisTypeaheadComboboxFooter.defaultProps = {
  _TYPE: 'TypeaheadComboboxFooter',
};

export const TypeaheadComboboxFooter = DuboisTypeaheadComboboxFooter;
