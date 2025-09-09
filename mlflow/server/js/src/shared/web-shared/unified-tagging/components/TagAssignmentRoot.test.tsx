import { render, renderHook, screen } from '@testing-library/react';
import type { UseFormReturn } from 'react-hook-form';
import { FormProvider, useForm } from 'react-hook-form';
import { IntlProvider } from 'react-intl';

import { TagAssignmentRoot } from './TagAssignmentRoot';
import { useTagAssignmentForm } from '../hooks/useTagAssignmentForm';

describe('TagAssignmentRoot', () => {
  function TestComponent({ children, form }: { children: React.ReactNode; form?: UseFormReturn }) {
    const tagsForm = useTagAssignmentForm({
      name: 'tags',
      emptyValue: { key: '', value: undefined },
      keyProperty: 'key',
      valueProperty: 'value',
      form,
    });

    return <TagAssignmentRoot {...tagsForm}>{children}</TagAssignmentRoot>;
  }
  it('should throw an error when used without a context or form prop', () => {
    expect(() =>
      render(
        <IntlProvider locale="en">
          <TestComponent>
            <div>child</div>
          </TestComponent>
        </IntlProvider>,
      ),
    ).toThrow('Nest your component on a FormProvider or pass a form prop');
  });

  it('should render child correctly if form prop is passed', () => {
    const { result } = renderHook(() => useForm());
    render(
      <IntlProvider locale="en">
        <TestComponent form={result.current}>
          <div>child</div>
        </TestComponent>
      </IntlProvider>,
    );

    expect(screen.getByText('child')).toBeInTheDocument();
  });

  it('should render child correctly if inside a form provider', () => {
    const { result } = renderHook(() => useForm());
    render(
      <IntlProvider locale="en">
        <FormProvider {...result.current}>
          <TestComponent>
            <div>child</div>
          </TestComponent>
        </FormProvider>
      </IntlProvider>,
    );

    expect(screen.getByText('child')).toBeInTheDocument();
  });
});
