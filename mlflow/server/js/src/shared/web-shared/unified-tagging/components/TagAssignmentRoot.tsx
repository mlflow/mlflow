import invariant from 'invariant';
import type { ArrayPath, FieldArray, FieldValues } from 'react-hook-form';
import { FormProvider, useFormContext } from 'react-hook-form';

import { TagAssignmentRowContainer } from './TagAssignmentUI/TagAssignmentRowContainer';
import { TagAssignmentContextProvider } from '../context/TagAssignmentContextProvider';
import type { UseTagAssignmentFormReturn } from '../hooks/useTagAssignmentForm';

export function TagAssignmentRoot<
  T extends FieldValues = FieldValues,
  K extends ArrayPath<T> = ArrayPath<T>,
  V extends FieldArray<T, K> = FieldArray<T, K>,
>({ children, ...props }: { children: React.ReactNode } & UseTagAssignmentFormReturn<T, K, V>) {
  const formCtx = useFormContext();

  const Component = (
    <TagAssignmentContextProvider {...props}>
      <TagAssignmentRowContainer>{children}</TagAssignmentRowContainer>
    </TagAssignmentContextProvider>
  );

  if (formCtx) {
    return Component;
  }

  invariant(props.form, 'Nest your component on a FormProvider or pass a form prop');

  return <FormProvider {...props.form}>{Component}</FormProvider>;
}
