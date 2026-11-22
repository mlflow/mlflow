import { useState } from 'react';
import { describe, expect, jest, test } from '@jest/globals';
import { act, fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { TraceFilterButton, type TraceFilterButtonProps } from './TraceFilterButton';
import { FilterOp, type TraceFilterModel } from './filterModel';

const duration = { field: 'duration', operator: FilterOp.GREATER_THAN, value: '1000' };
const format = (model: TraceFilterModel) => model.map(({ value }) => `duration > ${value}`).join(' AND ');
const parse = (value: string): TraceFilterModel | undefined =>
  /^duration > \d+$/.test(value) ? [{ ...duration, value: value.slice('duration > '.length) }] : undefined;
const wrapper = ({ children }: { children: React.ReactNode }) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>{children}</DesignSystemProvider>
  </IntlProvider>
);
const base: TraceFilterButtonProps = {
  fields: [{ id: 'duration', label: 'Duration', operators: [FilterOp.GREATER_THAN], valueInput: 'number' }],
  filterModel: [duration],
  onChange: jest.fn(),
  onClearAll: jest.fn(),
  activeCount: 1,
};
const sql = { value: 'duration > 1000', onChange: jest.fn(), formatBasic: format, parseBasic: parse };
const openSql = async () => {
  await userEvent.click(screen.getByRole('button', { name: 'Filters' }));
  await userEvent.click(screen.getByRole('tab', { name: 'SQL' }));
};
const editor = () => screen.getByRole('textbox', { name: 'Trace filter predicate' });

describe('trace SQL filter editor', () => {
  test('round trips a Basic draft and applies the converted SQL value', async () => {
    const onChange = jest.fn();
    render(<TraceFilterButton {...base} onChange={onChange} sqlFilter={sql} />, { wrapper });
    await openSql();
    expect(editor()).toHaveValue('duration > 1000');
    fireEvent.change(editor(), { target: { value: 'duration > 2000' } });
    await userEvent.click(screen.getByRole('tab', { name: 'Basic' }));
    expect(screen.getByRole('spinbutton', { name: 'Filter value' })).toHaveValue(2000);
    await userEvent.click(screen.getByRole('button', { name: 'Apply filters' }));
    expect(sql.onChange).toHaveBeenCalledWith('duration > 2000');
    expect(onChange).not.toHaveBeenCalled();
  });

  test('keeps unsupported predicates editable instead of losing them in Basic', async () => {
    render(<TraceFilterButton {...base} sqlFilter={sql} />, { wrapper });
    await openSql();
    fireEvent.change(editor(), { target: { value: "span.name RLIKE 'checkout.*'" } });
    await userEvent.click(screen.getByRole('tab', { name: 'Basic' }));
    expect(screen.getByRole('button', { name: 'Apply filters' })).toBeDisabled();
    expect(screen.getByRole('alert')).toHaveTextContent('cannot be represented');
    await userEvent.click(screen.getByRole('tab', { name: 'SQL' }));
    expect(editor()).toHaveValue("span.name RLIKE 'checkout.*'");
  });

  test('edits only the SQL chip through its scoped callback', async () => {
    const onClauseChange = jest.fn();
    render(
      <TraceFilterButton
        {...base}
        sqlFilter={{ ...sql, appliedClause: { value: "span.name IN ('a', 'b')", onChange: onClauseChange } }}
        renderTrigger={(open) => <button onClick={() => open('sql')}>Edit SQL chip</button>}
      />,
      { wrapper },
    );
    await userEvent.click(screen.getByRole('button', { name: 'Edit SQL chip' }));
    expect(await screen.findByRole('textbox', { name: 'Trace filter predicate' })).toHaveValue(
      "span.name IN ('a', 'b')",
    );
    fireEvent.change(editor(), { target: { value: "span.name IN ('a', 'b') AND tags.env = 'qa'" } });
    await userEvent.click(screen.getByRole('button', { name: 'Apply filter' }));
    expect(onClauseChange).toHaveBeenCalledWith("span.name IN ('a', 'b') AND tags.env = 'qa'");
  });

  test('uses the scoped replacement when SQL editing one Basic chip', async () => {
    const replaceClause = jest.fn((_index: number, _value: string) => 'duration > 2000');
    render(
      <TraceFilterButton
        {...base}
        sqlFilter={{ ...sql, replaceClause }}
        renderTrigger={(open) => <button onClick={() => open(0)}>Edit duration</button>}
      />,
      { wrapper },
    );
    await userEvent.click(screen.getByRole('button', { name: 'Edit duration' }));
    await userEvent.click(await screen.findByRole('tab', { name: 'SQL' }));
    fireEvent.change(editor(), { target: { value: 'duration > 2000' } });
    await userEvent.click(screen.getByRole('button', { name: 'Apply filter' }));
    expect(replaceClause).toHaveBeenCalledWith(0, 'duration > 2000');
  });

  test('restores a rejected draft but leaves a newer draft alone', async () => {
    let reject: () => void = () => {};
    const Harness = () => {
      const [value, setValue] = useState('');
      const [error, setError] = useState<Error>();
      reject = () => setError(new Error('Unsupported predicate'));
      return <TraceFilterButton {...base} filterModel={[]} sqlFilter={{ ...sql, value, onChange: setValue, error }} />;
    };
    render(<Harness />, { wrapper });
    await openSql();
    fireEvent.change(editor(), { target: { value: 'SELECT * FROM traces' } });
    await userEvent.click(screen.getByRole('button', { name: 'Apply filter' }));
    act(reject);
    expect(await screen.findByRole('textbox', { name: 'Trace filter predicate' })).toHaveValue('SELECT * FROM traces');
    expect(screen.getByRole('alert')).toHaveTextContent('Unsupported predicate');
    fireEvent.change(editor(), { target: { value: 'duration > 2000' } });
    act(reject);
    expect(editor()).toHaveValue('duration > 2000');
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });
});
