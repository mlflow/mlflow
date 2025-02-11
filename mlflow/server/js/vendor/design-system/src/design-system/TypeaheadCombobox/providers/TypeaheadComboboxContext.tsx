import type { ExtendedRefs } from '@floating-ui/react';
import { createContext, useState } from 'react';

import type { DesignSystemEventProviderAnalyticsEventTypes } from '../../DesignSystemEventProvider/DesignSystemEventProvider';
import type { AnalyticsEventProps } from '../../types';

export interface TypeaheadComboboxContextType
  extends AnalyticsEventProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange> {
  isInsideTypeaheadCombobox: boolean;
  multiSelect?: boolean;
  floatingUiRefs?: ExtendedRefs<Element>;
  floatingStyles?: React.CSSProperties;
  inputWidth?: number;
  setInputWidth?: (width: number) => void;
}

const typeaheadComboboxContextDefaults: TypeaheadComboboxContextType = {
  componentId: 'codegen_design-system_src_design-system_typeaheadcombobox_providers_typeaheadcomboboxcontext.tsx_17',
  isInsideTypeaheadCombobox: false,
  multiSelect: false,
};

export const TypeaheadComboboxContext = createContext<TypeaheadComboboxContextType>(typeaheadComboboxContextDefaults);

export const TypeaheadComboboxContextProvider = ({
  children,
  value,
}: {
  children: JSX.Element;
  value: TypeaheadComboboxContextType;
}) => {
  const [inputWidth, setInputWidth] = useState<number>();

  return (
    <TypeaheadComboboxContext.Provider value={{ ...value, setInputWidth, inputWidth }}>
      {children}
    </TypeaheadComboboxContext.Provider>
  );
};
