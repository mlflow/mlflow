import Plotly from 'plotly.js/lib/core';
import bar from 'plotly.js/lib/bar';
import box from 'plotly.js/lib/box';
import contour from 'plotly.js/lib/contour';
import parcoords from 'plotly.js/lib/parcoords';
import scatter from 'plotly.js/lib/scatter';
import scattergl from 'plotly.js/lib/scattergl';
import createPlotlyComponent from 'react-plotly.js/factory';

// Keep this list aligned with the trace types rendered in MLflow charts.
Plotly.register([scatter, scattergl, bar, box, contour, parcoords]);

export { Plotly };
export default createPlotlyComponent(Plotly);
