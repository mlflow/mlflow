import React from 'react';
import { ErrorView } from './ErrorView';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { MemoryRouter } from '../utils/RoutingUtils';

describe('ErrorView', () => {
  test('should render 400', () => {
    renderWithIntl(
      <MemoryRouter>
        <ErrorView statusCode={400} fallbackHomePageReactRoute="/path/to" />
      </MemoryRouter>,
    );

    const errorImage = screen.getByRole('img');
    expect(errorImage).toBeInTheDocument();
    expect(errorImage).toHaveAttribute('alt', '400 Bad Request');

    const title = screen.getByRole('heading', { level: 1 });
    expect(title).toBeInTheDocument();
    expect(title).toHaveTextContent('Bad Request');

    const subtitle = screen.getByRole('heading', { level: 2 });
    expect(subtitle).toBeInTheDocument();
    expect(subtitle).toHaveTextContent('Go back to');

    const link = screen.getByRole('link');
    expect(link).toBeInTheDocument();
    expect(link).toHaveAttribute('href', '/path/to');
  });

  it('should render 404', () => {
    renderWithIntl(
      <MemoryRouter>
        <ErrorView statusCode={404} fallbackHomePageReactRoute="/path/to" />
      </MemoryRouter>,
    );

    const errorImage = screen.getByRole('img');
    expect(errorImage).toBeInTheDocument();
    expect(errorImage).toHaveAttribute('alt', '404 Not Found');

    const title = screen.getByRole('heading', { level: 1 });
    expect(title).toBeInTheDocument();
    expect(title).toHaveTextContent('Page Not Found');

    const subtitle = screen.getByRole('heading', { level: 2 });
    expect(subtitle).toBeInTheDocument();
    expect(subtitle).toHaveTextContent('Go back to');

    const link = screen.getByRole('link');
    expect(link).toBeInTheDocument();
    expect(link).toHaveAttribute('href', '/path/to');
  });

  test('should render 404 with sub message', () => {
    renderWithIntl(
      <MemoryRouter>
        <ErrorView statusCode={404} fallbackHomePageReactRoute="/path/to" subMessage="sub message" />
      </MemoryRouter>,
    );

    const errorImage = screen.getByRole('img');
    expect(errorImage).toBeInTheDocument();
    expect(errorImage).toHaveAttribute('alt', '404 Not Found');

    const title = screen.getByRole('heading', { level: 1 });
    expect(title).toBeInTheDocument();
    expect(title).toHaveTextContent('Page Not Found');

    const subtitle = screen.getByRole('heading', { level: 2 });
    expect(subtitle).toBeInTheDocument();
    expect(subtitle).toHaveTextContent('sub message, go back to ');

    const link = screen.getByRole('link');
    expect(link).toBeInTheDocument();
    expect(link).toHaveAttribute('href', '/path/to');
  });
});
