import { createContext } from 'react';

export interface SelectContextType {
  isSelect: boolean;
  placeholder?: string;
}

const selectContextDefaults: SelectContextType = {
  isSelect: false,
};

export const SelectContext = createContext<SelectContextType>(selectContextDefaults);

export const SelectContextProvider = ({ children, value }: { children: JSX.Element; value: SelectContextType }) => {
  return <SelectContext.Provider value={value}>{children}</SelectContext.Provider>;
};
