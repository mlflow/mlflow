import { fireEvent, render, screen } from '@testing-library/react';
import { LineSmoothSlider } from './LineSmoothSlider';
import { useState } from 'react';
import userEvent from '@testing-library/user-event';

describe('LineSmoothSlider', () => {
  const TestComponent = ({ marks }: { marks?: Record<string, any> }) => {
    const [value, setValue] = useState(50);
    return (
      <>
        <LineSmoothSlider min={0} max={100} value={value} onChange={setValue} marks={marks} />
        current value: {value}
      </>
    );
  };

  test('should change value when slider is used', async () => {
    render(<TestComponent />);

    expect(screen.getByText('current value: 50')).toBeInTheDocument();

    fireEvent.keyDown(screen.getByRole('slider'), { key: 'ArrowRight' });
    expect(screen.getByText('current value: 51')).toBeInTheDocument();

    fireEvent.keyDown(screen.getByRole('slider'), { key: 'End' });
    expect(screen.getByText('current value: 100')).toBeInTheDocument();

    fireEvent.keyDown(screen.getByRole('slider'), { key: 'Home' });
    expect(screen.getByText('current value: 0')).toBeInTheDocument();
  });

  test('should change value when input value is changed with respect to boundaries', async () => {
    render(<TestComponent />);

    expect(screen.getByText('current value: 50')).toBeInTheDocument();

    await userEvent.clear(screen.getByRole('spinbutton'));
    await userEvent.type(screen.getByRole('spinbutton'), '25');
    fireEvent.blur(screen.getByRole('spinbutton'));
    expect(screen.getByText('current value: 25')).toBeInTheDocument();

    await userEvent.clear(screen.getByRole('spinbutton'));
    await userEvent.type(screen.getByRole('spinbutton'), '333');
    fireEvent.blur(screen.getByRole('spinbutton'));
    expect(screen.getByText('current value: 100')).toBeInTheDocument();
  });

  test('should respect marks when changing the value', async () => {
    const marks = { '0': '0', '30': '30', '50': '50', '100': '100' };
    render(<TestComponent marks={marks} />);

    expect(screen.getByText('current value: 50')).toBeInTheDocument();

    await userEvent.clear(screen.getByRole('spinbutton'));
    await userEvent.type(screen.getByRole('spinbutton'), '25');
    fireEvent.blur(screen.getByRole('spinbutton'));
    expect(screen.getByText('current value: 30')).toBeInTheDocument();

    await userEvent.clear(screen.getByRole('spinbutton'));
    await userEvent.type(screen.getByRole('spinbutton'), '888');
    fireEvent.blur(screen.getByRole('spinbutton'));

    expect(screen.getByText('current value: 100')).toBeInTheDocument();

    fireEvent.keyDown(screen.getByRole('slider'), { key: 'ArrowLeft' });

    expect(screen.getByText('current value: 50')).toBeInTheDocument();

    fireEvent.keyDown(screen.getByRole('slider'), { key: 'ArrowLeft' });

    expect(screen.getByText('current value: 30')).toBeInTheDocument();
  });
});
