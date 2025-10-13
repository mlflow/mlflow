import React from 'react';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { GetStarted } from './GetStarted';
import { homeQuickActions } from '../quick-actions';

type FormattedMessageReactElement = React.ReactElement<{ defaultMessage: string }>;

const isFormattedMessageElement = (node: React.ReactNode): node is FormattedMessageReactElement =>
  React.isValidElement(node) && typeof (node.props as { defaultMessage?: unknown }).defaultMessage === 'string';

const getQuickActionDefaultMessage = (title: React.ReactNode): string => {
  if (isFormattedMessageElement(title)) {
    return title.props.defaultMessage;
  }
  throw new Error('Expected quick action title to be a FormattedMessage element');
};

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
      const defaultMessage = getQuickActionDefaultMessage(action.title);
      const element = screen.getByText(defaultMessage);
      expect(element).toBeInTheDocument();
      expect(element.closest('a')).not.toBeNull();
    });
  });

  it('renders quick actions as external links when no handler is provided', () => {
    renderWithDesignSystem(<GetStarted />);

    homeQuickActions.forEach((action) => {
      const defaultMessage = getQuickActionDefaultMessage(action.title);
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

    const logTracesAction = homeQuickActions.find((action) => action.id === 'log-traces');
    if (!logTracesAction) {
      throw new Error('Log traces quick action is not defined');
    }

    const logTracesButton = screen.getByRole('button', {
      name: new RegExp(getQuickActionDefaultMessage(logTracesAction.title), 'i'),
    });

    await userEvent.click(logTracesButton);
    expect(onLogTracesClick).toHaveBeenCalled();

    homeQuickActions
      .filter((action) => action.id !== 'log-traces')
      .forEach((action) => {
        const link = screen
          .getByText(getQuickActionDefaultMessage(action.title))
          .closest('a') as HTMLAnchorElement | null;

        expect(link).not.toBeNull();
        expect(link).toHaveAttribute('href', action.link);
      });
  });
});
