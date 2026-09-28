import { describe, it, expect, jest } from '@jest/globals';
import { render, screen, fireEvent } from '@testing-library/react';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from 'react-intl';
import { IconEditor } from './IconEditor';
import type { MCPIcon } from '../types';

const renderEditor = (props: Partial<React.ComponentProps<typeof IconEditor>> = {}) => {
  const defaultProps = { icons: [] as MCPIcon[], onChange: jest.fn() };
  return render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <IconEditor {...defaultProps} {...props} />
      </DesignSystemProvider>
    </IntlProvider>,
  );
};

describe('IconEditor:interaction', () => {
  it('renders draft row with disabled + button when URL is empty', () => {
    renderEditor();
    expect(screen.getByPlaceholderText('https://example.com/icon.svg')).toBeInTheDocument();
    expect(screen.getByLabelText('Add icon')).toBeDisabled();
  });

  it('adds icon and clears draft when + is clicked', () => {
    const onChange = jest.fn();
    renderEditor({ onChange });

    fireEvent.change(screen.getByPlaceholderText('https://example.com/icon.svg'), {
      target: { value: 'https://example.com/icon.svg' },
    });
    fireEvent.click(screen.getByLabelText('Add icon'));

    expect(onChange).toHaveBeenCalledWith([{ src: 'https://example.com/icon.svg' }]);
  });

  it('renders confirmed rows with X and draft row with +', () => {
    renderEditor({
      icons: [{ src: 'https://example.com/light.svg', theme: 'light' }],
    });

    const inputs = screen.getAllByPlaceholderText('https://example.com/icon.svg');
    expect(inputs).toHaveLength(2);
    expect(inputs[0]).toHaveValue('https://example.com/light.svg');
    expect(inputs[1]).toHaveValue('');
    expect(screen.getByLabelText('Remove icon')).toBeInTheDocument();
    expect(screen.getByLabelText('Add icon')).toBeInTheDocument();
  });

  it('calls onChange on blur after editing a confirmed row URL', () => {
    const onChange = jest.fn();
    renderEditor({ icons: [{ src: 'https://example.com/old.svg' }], onChange });

    const input = screen.getAllByPlaceholderText('https://example.com/icon.svg')[0];
    fireEvent.change(input, { target: { value: 'https://example.com/new.svg' } });
    expect(onChange).not.toHaveBeenCalled();
    fireEvent.blur(input);

    expect(onChange).toHaveBeenCalledWith([{ src: 'https://example.com/new.svg' }]);
  });

  it('calls onChange when removing a confirmed icon', () => {
    const onChange = jest.fn();
    renderEditor({
      icons: [
        { src: 'https://example.com/a.svg', theme: 'light' },
        { src: 'https://example.com/b.svg', theme: 'dark' },
      ],
      onChange,
    });

    fireEvent.click(screen.getAllByLabelText('Remove icon')[0]);
    expect(onChange).toHaveBeenCalledWith([{ src: 'https://example.com/b.svg', theme: 'dark' }]);
  });
});

describe('IconEditor:preview fallback chain', () => {
  it('explicit light + serverJson dark: light preview shows explicit, dark shows serverJson', () => {
    const { container } = renderEditor({
      icons: [{ src: 'https://example.com/explicit-light.svg', theme: 'light' }],
      serverJsonIcons: [{ src: 'https://example.com/sj-dark.svg', theme: 'dark' }],
    });

    const imgs = container.querySelectorAll('img');
    const srcs = Array.from(imgs).map((img) => img.getAttribute('src'));
    expect(srcs).toContain('https://example.com/explicit-light.svg');
    expect(srcs).toContain('https://example.com/sj-dark.svg');
  });

  it('explicit light only: light preview shows it, dark preview shows McpIcon', () => {
    const { container } = renderEditor({
      icons: [{ src: 'https://example.com/light.svg', theme: 'light' }],
    });

    const imgs = container.querySelectorAll('img');
    expect(imgs).toHaveLength(1);
    expect(imgs[0]).toHaveAttribute('src', 'https://example.com/light.svg');
    expect(container.querySelectorAll('svg').length).toBeGreaterThan(0);
  });

  it('explicit any icon: both previews show it', () => {
    const { container } = renderEditor({
      icons: [{ src: 'https://example.com/any.svg' }],
    });

    const imgs = container.querySelectorAll('img');
    expect(imgs).toHaveLength(2);
    expect(imgs[0]).toHaveAttribute('src', 'https://example.com/any.svg');
    expect(imgs[1]).toHaveAttribute('src', 'https://example.com/any.svg');
  });

  it('no icons and no serverJsonIcons: both previews show McpIcon fallback', () => {
    const { container } = renderEditor();

    expect(container.querySelector('img')).toBeNull();
    expect(container.querySelectorAll('svg').length).toBeGreaterThanOrEqual(2);
  });
});

describe('IconEditor:error + fallback on load failure', () => {
  it('shows error message and falls back to serverJsonIcons when explicit icon fails', () => {
    const { container } = renderEditor({
      icons: [{ src: 'https://example.com/broken.svg' }],
      serverJsonIcons: [{ src: 'https://example.com/sj-fallback.svg' }],
    });

    const imgs = container.querySelectorAll('img');
    fireEvent.error(imgs[0]);

    expect(screen.getByText('Image failed to load')).toBeInTheDocument();
    const updatedImgs = container.querySelectorAll('img');
    const srcs = Array.from(updatedImgs).map((img) => img.getAttribute('src'));
    expect(srcs).toContain('https://example.com/sj-fallback.svg');
  });

  it('shows error message and McpIcon when explicit icon fails and no serverJsonIcons', () => {
    const { container } = renderEditor({
      icons: [{ src: 'https://example.com/broken.svg' }],
    });

    const imgs = container.querySelectorAll('img');
    fireEvent.error(imgs[0]);

    expect(screen.getByText('Image failed to load')).toBeInTheDocument();
    expect(container.querySelectorAll('svg').length).toBeGreaterThan(0);
  });
});
