import { createContext } from 'react';

export interface DialogComboboxOptionListContextType {
  isInsideDialogComboboxOptionList?: boolean;
  lookAhead: string;
  setLookAhead: (lookAhead: string) => void;
}

export const DialogComboboxOptionListContext = createContext<DialogComboboxOptionListContextType>({
  isInsideDialogComboboxOptionList: false,
  lookAhead: '',
  setLookAhead: () => {},
});

export const DialogComboboxOptionListContextProvider = ({
  children,
  value,
}: {
  children: JSX.Element;
  value: DialogComboboxOptionListContextType;
}) => {
  return <DialogComboboxOptionListContext.Provider value={value}>{children}</DialogComboboxOptionListContext.Provider>;
};
