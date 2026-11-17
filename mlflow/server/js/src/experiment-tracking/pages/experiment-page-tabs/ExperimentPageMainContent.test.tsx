import { describe, expect, jest, test } from '@jest/globals';
import { useEffect } from 'react';
import { render, screen } from '@testing-library/react';
import { DesignSystemProvider } from '@databricks/design-system';
import { ExperimentPageMainContent } from './ExperimentPageMainContent';

describe('ExperimentPageMainContent', () => {
  test('groups shared breadcrumbs and routed tab content in the main landmark', () => {
    render(
      <DesignSystemProvider>
        <ExperimentPageMainContent breadcrumbs={<div>Breadcrumbs</div>} pageActions={<button>Page action</button>}>
          <div>Tab content</div>
        </ExperimentPageMainContent>
      </DesignSystemProvider>,
    );

    const main = screen.getByRole('main');
    expect(main).toContainElement(screen.getByText('Breadcrumbs'));
    expect(main).toContainElement(screen.getByText('Tab content'));
    expect(main).toContainElement(screen.getByRole('button', { name: 'Page action' }));
  });

  test('keeps the classic container without adding a landmark or rendering revamped headers', () => {
    render(
      <DesignSystemProvider>
        <ExperimentPageMainContent
          revamped={false}
          breadcrumbs={<div>Breadcrumbs</div>}
          pageActions={<button>Page action</button>}
          headerDescription={<div>Description</div>}
        >
          <div>Tab content</div>
        </ExperimentPageMainContent>
      </DesignSystemProvider>,
    );

    const wrapper = screen.getByText('Tab content').parentElement?.parentElement;
    expect(wrapper?.tagName).toBe('DIV');
    expect(screen.queryByRole('main')).not.toBeInTheDocument();
    expect(screen.queryByText('Breadcrumbs')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Page action' })).not.toBeInTheDocument();
    expect(screen.queryByText('Description')).not.toBeInTheDocument();
  });

  test('leaves out the header row without breadcrumbs or actions, without remounting the content', () => {
    const onMount = jest.fn();
    const TabContent = () => {
      useEffect(() => {
        onMount();
      }, []);
      return <div>Tab content</div>;
    };
    const renderContent = (breadcrumbs?: React.ReactNode) => (
      <DesignSystemProvider>
        <ExperimentPageMainContent breadcrumbs={breadcrumbs}>
          <TabContent />
        </ExperimentPageMainContent>
      </DesignSystemProvider>
    );
    const { rerender } = render(renderContent(<div>Breadcrumbs</div>));

    rerender(renderContent());
    expect(screen.queryByText('Breadcrumbs')).not.toBeInTheDocument();
    expect(screen.getByRole('main').children).toHaveLength(1);

    rerender(renderContent(<div>Breadcrumbs</div>));
    expect(screen.getByText('Breadcrumbs')).toBeInTheDocument();
    expect(onMount).toHaveBeenCalledTimes(1);
  });
});
