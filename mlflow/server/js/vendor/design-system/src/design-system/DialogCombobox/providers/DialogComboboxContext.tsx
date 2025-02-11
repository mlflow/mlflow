import { createContext } from 'react';

import type { DesignSystemEventProviderAnalyticsEventTypes } from '../../DesignSystemEventProvider/DesignSystemEventProvider';
import type { AnalyticsEventValueChangeNoPiiFlagProps } from '../../types';

export interface DialogComboboxContextType
  extends AnalyticsEventValueChangeNoPiiFlagProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange> {
  id?: string;
  label?: string | React.ReactNode;
  value: string[];
  isInsideDialogCombobox: boolean;
  multiSelect?: boolean;
  setValue: (value: string[]) => void;
  setIsControlled: (isControlled: boolean) => void;
  stayOpenOnSelection?: boolean;
  isOpen?: boolean;
  setIsOpen: (isOpen: boolean) => void;
  emptyText?: string;
  contentWidth: number | string | undefined;
  setContentWidth: (width: number | string | undefined) => void;
  textOverflowMode: 'ellipsis' | 'multiline';
  setTextOverflowMode: (mode: 'ellipsis' | 'multiline') => void;
  scrollToSelectedElement: boolean;
  rememberLastScrollPosition: boolean;
}

const dialogComboboxContextDefaults: DialogComboboxContextType = {
  componentId: 'codegen_design-system_src_design-system_dialogcombobox_providers_dialogcomboboxcontext.tsx_27',
  id: '',
  label: '',
  value: [],
  isInsideDialogCombobox: false,
  multiSelect: false,
  setValue: () => {},
  setIsControlled: () => {},
  stayOpenOnSelection: false,
  isOpen: false,
  setIsOpen: () => {},
  contentWidth: undefined,
  setContentWidth: () => {},
  textOverflowMode: 'multiline',
  setTextOverflowMode: () => {},
  scrollToSelectedElement: true,
  rememberLastScrollPosition: false,
};

export const DialogComboboxContext = createContext<DialogComboboxContextType>(dialogComboboxContextDefaults);

export const DialogComboboxContextProvider = ({
  children,
  value,
}: {
  children: JSX.Element;
  value: DialogComboboxContextType;
}) => {
  return <DialogComboboxContext.Provider value={value}>{children}</DialogComboboxContext.Provider>;
};
