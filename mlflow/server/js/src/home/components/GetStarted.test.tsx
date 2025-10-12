import React from 'react';
import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { GetStarted } from './GetStarted';
import { homeQuickActions } from '../quick-actions';

describe('GetStarted', () => {
  it('renders header and all quick actions', () => {
    renderWithDesignSystem(<GetStarted />);

    expect(
      screen.getByRole('heading', {
        level: 3,
        name: 'Get started',
      }),
    ).toBeInTheDocument();

    homeQuickActions.forEach((action) => {
      const defaultMessage = (action.title as React.ReactElement).props.defaultMessage;
      const element = screen.getByText(defaultMessage);
      expect(element).toBeInTheDocument();
      expect(element.closest('a')).not.toBeNull();
    });
  });

  it('renders routing links with proper attributes', () => {
    renderWithDesignSystem(<GetStarted />);

    homeQuickActions.forEach((action) => {
      const defaultMessage = (action.title as React.ReactElement).props.defaultMessage;
      const linkElement = screen.getByText(defaultMessage).closest('a') as HTMLAnchorElement | null;

      expect(linkElement).not.toBeNull();
      expect(linkElement).toHaveAttribute('href', action.link);
      expect(linkElement).toHaveAttribute('target', '_blank');
      expect(linkElement).toHaveAttribute('rel', 'noopener noreferrer');
    });
  });
});
