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
      const defaultMessage = (item.title as React.ReactElement).props.defaultMessage;
      const element = screen.getByText(defaultMessage);
      expect(element).toBeInTheDocument();
      expect(element.closest('a')).not.toBeNull();
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
      const defaultMessage = (item.title as React.ReactElement).props.defaultMessage;
      const linkElement = screen.getByText(defaultMessage).closest('a') as HTMLAnchorElement | null;

      expect(linkElement).not.toBeNull();
      expect(linkElement).toHaveAttribute('href', item.link);
      expect(linkElement).toHaveAttribute('target', '_blank');
      expect(linkElement).toHaveAttribute('rel', 'noopener noreferrer');
    });
  });
});
