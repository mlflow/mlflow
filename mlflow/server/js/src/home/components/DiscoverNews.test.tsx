import React from 'react';
import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { DiscoverNews } from './DiscoverNews';
import { homeNewsItems } from '../news-items';

describe('DiscoverNews', () => {
  it('renders section header and all news links', () => {
    renderWithDesignSystem(<DiscoverNews />);

    expect(
      screen.getByRole('heading', {
        level: 3,
        name: 'Discover new features',
      }),
    ).toBeInTheDocument();

    homeNewsItems.forEach((item) => {
      const title = screen.getByText(item.title.props.defaultMessage);
      expect(title).toBeInTheDocument();
      expect(title.closest('a')).not.toBeNull();
    });
  });

  it('shows link to announcements page', () => {
    renderWithDesignSystem(<DiscoverNews />);

    expect(
      screen.getByRole('link', {
        name: 'View all',
      }),
    ).toBeInTheDocument();
  });

  it('renders news links with correct attributes', () => {
    renderWithDesignSystem(<DiscoverNews />);

    homeNewsItems.forEach((item) => {
      const link = screen
        .getByText(item.title.props.defaultMessage)
        .closest('a') as HTMLAnchorElement | null;

      expect(link).not.toBeNull();
      expect(link).toHaveAttribute('href', item.link);
      expect(link).toHaveAttribute('target', '_blank');
      expect(link).toHaveAttribute('rel', 'noopener noreferrer');
    });
  });
});
