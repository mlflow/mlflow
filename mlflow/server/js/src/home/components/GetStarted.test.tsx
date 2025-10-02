import React from 'react';
import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { GetStarted } from './GetStarted';
import { homeQuickActions } from '../quick-actions';
import { MemoryRouter } from '../../common/utils/RoutingUtils';

describe('GetStarted', () => {
  it('renders header and all quick actions', () => {
    const { container } = renderWithDesignSystem(<GetStarted />);

    expect(
      screen.getByRole('heading', {
        level: 3,
        name: 'Get started',
      }),
    ).toBeInTheDocument();

    homeQuickActions.forEach((action) => {
      expect(container.querySelector(`[data-component-id="${action.componentId}"]`)).toBeInTheDocument();
    });
  });

  it('renders routing links with proper attributes', () => {
    renderWithDesignSystem(
      <MemoryRouter>
        <GetStarted />
      </MemoryRouter>,
    );

    const externalAction = homeQuickActions.find((action) => action.link.type === 'external');
    const internalAction = homeQuickActions.find((action) => action.link.type === 'internal');

    if (!externalAction || !internalAction) {
      throw new Error('Expected both external and internal actions to be defined');
    }

    const externalLink = screen.getByRole('link', {
      name: externalAction.title.props.defaultMessage,
    }) as HTMLAnchorElement;
    expect(externalLink).toHaveAttribute('href', externalAction.link.href);
    expect(externalLink).toHaveAttribute('target', '_blank');

    const internalLink = screen.getByRole('link', {
      name: internalAction.title.props.defaultMessage,
    }) as HTMLAnchorElement;
    expect(internalLink).toHaveAttribute('href', internalAction.link.to);
  });
});
