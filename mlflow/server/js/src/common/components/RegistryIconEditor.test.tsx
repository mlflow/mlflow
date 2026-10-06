import { describe, expect, it, jest } from '@jest/globals';
import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider, PuzzleIcon } from '@databricks/design-system';

import type { RegistryIconImage } from '../utils/registryIcons';
import { RegistryIconEditor } from './RegistryIconEditor';

const LIGHT = { src: 'https://example.com/light.svg', theme: 'light' };
const DARK = { src: 'https://example.com/dark.svg' };

const ControlledEditor = ({
  initial,
  onChange,
}: {
  initial: RegistryIconImage[];
  onChange: (icons: RegistryIconImage[]) => void;
}) => {
  const [icons, setIcons] = useState(initial);
  return (
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <RegistryIconEditor
          icons={icons}
          defaultIcon={<PuzzleIcon />}
          componentId="test.icon_editor"
          onChange={(next) => {
            onChange(next);
            setIcons(next);
          }}
        />
      </DesignSystemProvider>
    </IntlProvider>
  );
};

describe('RegistryIconEditor', () => {
  it('gives each existing icon URL input its own accessible name', () => {
    render(<ControlledEditor initial={[LIGHT, DARK]} onChange={jest.fn()} />);

    expect(screen.getByRole('textbox', { name: 'Icon URL 1' })).toHaveValue('https://example.com/light.svg');
    expect(screen.getByRole('textbox', { name: 'Icon URL 2' })).toHaveValue('https://example.com/dark.svg');
  });

  it('removes an icon whose URL was just edited', async () => {
    const onChange = jest.fn<(icons: RegistryIconImage[]) => void>();
    render(<ControlledEditor initial={[LIGHT, DARK]} onChange={onChange} />);

    await userEvent.type(screen.getByRole('textbox', { name: 'Icon URL 1' }), '?v=2');
    await userEvent.click(screen.getAllByRole('button', { name: 'Remove icon' })[0]);

    expect(onChange).toHaveBeenLastCalledWith([DARK]);
    expect(screen.getAllByRole('textbox', { name: /^Icon URL \d$/ })).toHaveLength(1);
    expect(screen.getByRole('textbox', { name: 'Icon URL 1' })).toHaveValue(DARK.src);
  });

  it('reports an icon URL that fails to load', () => {
    render(<ControlledEditor initial={[DARK]} onChange={jest.fn()} />);

    // Both theme previews resolve to the theme-agnostic icon.
    fireEvent.error(document.querySelector('img') as HTMLImageElement);

    expect(screen.getByText('Image failed to load')).toBeInTheDocument();
    expect(document.querySelector('img')).toBeNull();
  });

  it('loads a failed URL again after it is edited away and back', async () => {
    render(<ControlledEditor initial={[DARK]} onChange={jest.fn()} />);
    fireEvent.error(document.querySelector('img') as HTMLImageElement);
    expect(screen.getByText('Image failed to load')).toBeInTheDocument();

    const input = screen.getByRole('textbox', { name: 'Icon URL 1' });
    await userEvent.clear(input);
    await userEvent.type(input, 'https://example.com/other.svg');
    await userEvent.tab();
    await userEvent.clear(input);
    await userEvent.type(input, DARK.src);
    await userEvent.tab();

    expect(screen.queryByText('Image failed to load')).not.toBeInTheDocument();
    expect(document.querySelector(`img[src="${DARK.src}"]`)).not.toBeNull();
  });
});
