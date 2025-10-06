import React from 'react';
import userEvent from '@testing-library/user-event';
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

  it('renders quick actions as external links when no handler is provided', () => {
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

  it('invokes handler when log traces quick action is clicked', async () => {
    const onLogTracesClick = jest.fn();
    renderWithDesignSystem(<GetStarted onLogTracesClick={onLogTracesClick} />);

    const logTracesButton = screen.getByRole('button', {
      name: homeQuickActions[0].title.props.defaultMessage,
    });

    await userEvent.click(logTracesButton);
    expect(onLogTracesClick).toHaveBeenCalled();

    homeQuickActions
      .filter((action) => action.id !== 'log-traces')
      .forEach((action) => {
        const link = screen.getByText(action.title.props.defaultMessage).closest('a') as HTMLAnchorElement | null;

        expect(link).not.toBeNull();
        expect(link).toHaveAttribute('href', action.link);
      });
  });
});
