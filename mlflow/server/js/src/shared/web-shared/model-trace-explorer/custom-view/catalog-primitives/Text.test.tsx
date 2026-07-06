import { describe, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';

import { Catalog, ComponentContext, ComponentModel, SurfaceModel } from '@a2ui/web_core/v0_9';
import { DesignSystemProvider } from '@databricks/design-system';

import { Text } from './Text';

const Wrapper = ({ children }: { children: React.ReactNode }) => (
  <DesignSystemProvider>{children}</DesignSystemProvider>
);

// Renders the Text implementation through a minimal A2UI surface so the generic
// binder resolves the component's props the same way the renderer does at runtime.
const renderText = (props: Record<string, unknown>) => {
  const catalog = new Catalog<any>('test', [], []);
  const surface = new SurfaceModel<any>('test-surface', catalog);
  surface.componentsModel.addComponent(new ComponentModel('c1', 'Text', props));
  const context = new ComponentContext(surface, 'c1', '/');
  return render(<Text.render context={context} buildChild={() => null} />, { wrapper: Wrapper });
};

describe('Text primitive', () => {
  it('renders body text', () => {
    renderText({ text: 'Hello World' });
    expect(screen.getByText('Hello World')).toBeInTheDocument();
  });

  it('renders a heading for the h1 variant', () => {
    renderText({ text: 'Big Title', variant: 'h1' });
    expect(screen.getByRole('heading', { level: 1, name: 'Big Title' })).toBeInTheDocument();
  });
});
