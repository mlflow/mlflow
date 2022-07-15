import { ClientSideRowModelModule } from '@ag-grid-community/client-side-row-model';
import { AgGridReact, AgGridReactProps, AgReactUiProps } from '@ag-grid-community/react/main';
import '@ag-grid-community/core/dist/styles/ag-grid.css';
import '@ag-grid-community/core/dist/styles/ag-theme-balham.css';
import { AgGridFontInjector } from './AgGridFontInjector';

/**
 * A local wrapper component that embeds imported AgGrid instance.
 * Extracted to a separate module to ensure that it will be in placed a single chunk.
 */
export const MLFlowAgGrid = (props: AgGridReactProps | AgReactUiProps) => (
  <>
    <AgGridFontInjector />
    <AgGridReact modules={[ClientSideRowModelModule]} {...props} />
  </>
);

export default MLFlowAgGrid;
