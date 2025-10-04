import React from 'react';
import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { DiscoverNews } from './DiscoverNews';
import { homeNewsItems } from '../news-items';

describe('DiscoverNews', () => {
  it('renders section header and all news links', () => {
    const { container } = renderWithDesignSystem(<DiscoverNews />);

    expect(
      screen.getByRole('heading', {
        level: 3,
        name: 'Discover new features',
      }),
    ).toBeInTheDocument();

    homeNewsItems.forEach((item) => {
      expect(container.querySelector(`[data-component-id="${item.componentId}"]`)).toBeInTheDocument();
    });
  });

  it('shows link to announcements page', () => {
    renderWithDesignSystem(<DiscoverNews />);

    expect(
      screen.getByRole('link', {
        name: '>>> See more announcements',
      }),
    ).toBeInTheDocument();
  });

  it('renders external links with correct attributes', () => {
    const { container } = renderWithDesignSystem(<DiscoverNews />);
    const externalItem = homeNewsItems.find((item) => item.link.type === 'external');

    if (!externalItem || externalItem.link.type !== 'external') {
      throw new Error('Expected at least one external news item');
    }

    const externalLink = container.querySelector(
      `[data-component-id="${externalItem.componentId}"]`,
    ) as HTMLAnchorElement | null;

    expect(externalLink).not.toBeNull();
    expect(externalLink).toHaveAttribute('href', externalItem.link.href);
    expect(externalLink).toHaveAttribute('target', '_blank');
    expect(externalLink).toHaveAttribute('rel', 'noopener noreferrer');
  });
});
