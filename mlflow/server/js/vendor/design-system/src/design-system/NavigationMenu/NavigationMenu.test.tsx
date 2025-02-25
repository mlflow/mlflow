import { act, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';

import { NavigationMenu } from '.';

describe('NavigationMenu', () => {
  const Component = () => {
    const [active, setActive] = useState(0);

    return (
      <NavigationMenu.Root>
        <NavigationMenu.List>
          {[...Array(3)].map((_o, idx) => (
            <NavigationMenu.Item key={idx} active={idx === active}>
              <a href={`#${idx}`} onClick={() => setActive(idx)}>{`Link ${idx}`}</a>
            </NavigationMenu.Item>
          ))}
        </NavigationMenu.List>
        {`Link ${active} Content`}
      </NavigationMenu.Root>
    );
  };

  it('renders a set of links', async () => {
    render(<Component />);

    const list = screen.getByRole('list');
    const links = within(list).getAllByRole('link');

    expect(links).toHaveLength(3);
    expect(links[0]).toHaveTextContent('Link 0');
    expect(links[1]).toHaveTextContent('Link 1');
    expect(links[2]).toHaveTextContent('Link 2');

    expect(links[0]).toHaveAttribute('aria-current', 'page');
    expect(screen.getByText('Link 0 Content')).toBeInTheDocument();
    expect(screen.queryByText('Link 1 Content')).not.toBeInTheDocument();

    await userEvent.click(links[1]);
    expect(links[1]).toHaveAttribute('aria-current', 'page');
    expect(screen.getByText('Link 1 Content')).toBeInTheDocument();
    expect(screen.queryByText('Link 0 Content')).not.toBeInTheDocument();
  });

  it('links are keyboard navigable', async () => {
    render(<Component />);

    const list = screen.getByRole('list');
    const links = within(list).getAllByRole('link');

    act(() => {
      links[0].focus();
    });

    await userEvent.keyboard('{ArrowRight}');
    expect(links[1]).toHaveFocus();
    await userEvent.keyboard('{Enter}');
    expect(links[1]).toHaveAttribute('aria-current', 'page');
    expect(screen.getByText('Link 1 Content')).toBeInTheDocument();

    await userEvent.keyboard('{ArrowLeft}');
    expect(links[0]).toHaveFocus();
    await userEvent.keyboard('{Enter}');
    expect(links[0]).toHaveAttribute('aria-current', 'page');
    expect(screen.getByText('Link 0 Content')).toBeInTheDocument();

    await userEvent.keyboard('{Tab}');
    expect(links[1]).toHaveFocus();
    await userEvent.keyboard('{Enter}');
    expect(links[1]).toHaveAttribute('aria-current', 'page');
    expect(screen.getByText('Link 1 Content')).toBeInTheDocument();

    await userEvent.keyboard('{Shift>}{Tab}');
    expect(links[0]).toHaveFocus();
    await userEvent.keyboard('{Enter}');
    expect(links[0]).toHaveAttribute('aria-current', 'page');
    expect(screen.getByText('Link 0 Content')).toBeInTheDocument();
  });
});
