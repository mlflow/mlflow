import { useContext } from 'react';

import { DialogComboboxContext } from '../providers/DialogComboboxContext';

export const useDialogComboboxContext = () => {
  return useContext(DialogComboboxContext);
};
