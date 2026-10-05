import { describe, expect, it, jest } from '@jest/globals';
import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';

import type { RegistryIcon } from '../types';
import { SkillIconEditor } from './SkillIconEditor';

const LIGHT = { src: 'https://example.com/light.svg', theme: 'light' };
const DARK = { src: 'https://example.com/dark.svg' };

const ControlledEditor = ({
  initial,
  onChange,
}: {
  initial: RegistryIcon[];
  onChange: (icons: RegistryIcon[]) => void;
}) => {
  const [icons, setIcons] = useState(initial);
  return (
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <SkillIconEditor
          icons={icons}
          onChange={(next) => {
            onChange(next);
            setIcons(next);
          }}
        />
      </DesignSystemProvider>
    </IntlProvider>
  );
};

describe('SkillIconEditor', () => {
  it('gives each existing icon URL input its own accessible name', () => {
    render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <SkillIconEditor
            icons={[{ src: 'https://example.com/light.svg', theme: 'light' }, { src: 'https://example.com/dark.svg' }]}
            onChange={jest.fn()}
          />
        </DesignSystemProvider>
      </IntlProvider>,
    );

    expect(screen.getByRole('textbox', { name: 'Icon URL 1' })).toHaveValue('https://example.com/light.svg');
    expect(screen.getByRole('textbox', { name: 'Icon URL 2' })).toHaveValue('https://example.com/dark.svg');
  });

  it('removes an icon whose URL was just edited', async () => {
    const onChange = jest.fn<(icons: RegistryIcon[]) => void>();
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
});
