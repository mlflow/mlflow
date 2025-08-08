import { FormProvider, useForm } from 'react-hook-form';

import { TagAssignmentContext } from '../context/TagAssignmentContextProvider';
import { useTagAssignmentForm } from '../hooks/useTagAssignmentForm';

interface TestFormI {
  tags: {
    key: string;
    value: string;
  }[];
}

interface TestTagAssignmentContextProviderProps extends Partial<ReturnType<typeof useTagAssignmentForm<TestFormI>>> {
  children: React.ReactNode;
}

export function TestTagAssignmentContextProvider({ children, ...props }: TestTagAssignmentContextProviderProps) {
  const form = useForm<TestFormI>();
  const tagForm = useTagAssignmentForm<TestFormI, 'tags', { key: string; value: string }>({
    form,
    name: 'tags',
    emptyValue: { key: '', value: '' },
    keyProperty: 'key',
    valueProperty: 'value',
  });
  return (
    <FormProvider {...form}>
      <TagAssignmentContext.Provider
        value={{
          ...(tagForm as any),
          ...props,
        }}
      >
        {children}
      </TagAssignmentContext.Provider>
    </FormProvider>
  );
}
