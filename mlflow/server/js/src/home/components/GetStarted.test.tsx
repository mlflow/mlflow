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
      const title = screen.getByText(action.title.props.defaultMessage);
      expect(title).toBeInTheDocument();
      expect(title.closest('a')).not.toBeNull();
    });
  });

  it('renders routing links with proper attributes', () => {
    renderWithDesignSystem(<GetStarted />);

    homeQuickActions.forEach((action) => {
      const link = screen
        .getByText(action.title.props.defaultMessage)
        .closest('a') as HTMLAnchorElement | null;

      expect(link).not.toBeNull();
      expect(link).toHaveAttribute('href', action.link);
      expect(link).toHaveAttribute('target', '_blank');
      expect(link).toHaveAttribute('rel', 'noopener noreferrer');
    });
  });
});
