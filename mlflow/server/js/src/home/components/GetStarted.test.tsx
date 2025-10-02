import React, { useEffect } from 'react';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { GetStarted } from './GetStarted';
import { homeQuickActions } from '../quick-actions';
import { HomePageViewStateProvider, useHomePageViewState } from '../HomePageViewStateContext';

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
    renderWithDesignSystem(
      <HomePageViewStateProvider>
        <GetStarted />
      </HomePageViewStateProvider>,
    );

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
      if (action.id === 'log-traces' || action.id === 'run-evaluation') {
        expect(element.closest('button')).not.toBeNull();
      } else {
        expect(element.closest('a')).not.toBeNull();
      }
    });
  });

  it('renders quick actions as buttons or links depending on configuration', () => {
    renderWithDesignSystem(
      <HomePageViewStateProvider>
        <GetStarted />
      </HomePageViewStateProvider>,
    );

    homeQuickActions.forEach((action) => {
      const defaultMessage = getQuickActionDefaultMessage(action.title);
      if (action.id === 'log-traces' || action.id === 'run-evaluation') {
        const buttonElement = screen.getByText(defaultMessage).closest('button');
        expect(buttonElement).not.toBeNull();
      } else {
        const linkElement = screen.getByText(defaultMessage).closest('a') as HTMLAnchorElement | null;
        expect(linkElement).not.toBeNull();
        expect(linkElement).toHaveAttribute('href', action.link);
        expect(linkElement).toHaveAttribute('target', '_blank');
        expect(linkElement).toHaveAttribute('rel', 'noopener noreferrer');
      }
    });
  });

  const DrawerObserver = ({
    onLogTracesOpen,
    onRunEvaluationOpen,
  }: {
    onLogTracesOpen?: jest.Mock;
    onRunEvaluationOpen?: jest.Mock;
  }) => {
    const { isLogTracesDrawerOpen, isRunEvaluationDrawerOpen } = useHomePageViewState();
    useEffect(() => {
      if (isLogTracesDrawerOpen) {
        onLogTracesOpen?.();
      }
    }, [isLogTracesDrawerOpen, onLogTracesOpen]);
    useEffect(() => {
      if (isRunEvaluationDrawerOpen) {
        onRunEvaluationOpen?.();
      }
    }, [isRunEvaluationDrawerOpen, onRunEvaluationOpen]);
    return null;
  };

  it('opens log traces drawer state when quick action is clicked', async () => {
    const onOpen = jest.fn();
    renderWithDesignSystem(
      <HomePageViewStateProvider>
        <DrawerObserver onLogTracesOpen={onOpen} />
        <GetStarted />
      </HomePageViewStateProvider>,
    );

    const logTracesAction = homeQuickActions.find((action) => action.id === 'log-traces');
    if (!logTracesAction) {
      throw new Error('Log traces quick action is not defined');
    }

    const logTracesButton = screen.getByRole('button', {
      name: new RegExp(getQuickActionDefaultMessage(logTracesAction.title), 'i'),
    });

    await userEvent.click(logTracesButton);
    expect(onOpen).toHaveBeenCalled();
  });

  it('opens run evaluation drawer state when quick action is clicked', async () => {
    const onOpen = jest.fn();
    renderWithDesignSystem(
      <HomePageViewStateProvider>
        <DrawerObserver onRunEvaluationOpen={onOpen} />
        <GetStarted />
      </HomePageViewStateProvider>,
    );

    const runEvaluationAction = homeQuickActions.find((action) => action.id === 'run-evaluation');
    if (!runEvaluationAction) {
      throw new Error('Run evaluation quick action is not defined');
    }

    const runEvaluationButton = screen.getByRole('button', {
      name: new RegExp(getQuickActionDefaultMessage(runEvaluationAction.title), 'i'),
    });

    await userEvent.click(runEvaluationButton);
    expect(onOpen).toHaveBeenCalled();
  });
});
