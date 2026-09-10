.. _python-api:

Python API
==========

The MLflow Python API is organized into the following modules. The most common functions are
exposed in the :py:mod:`mlflow` module, so we recommend starting there.

.. toctree::
  :glob:
  :maxdepth: 1

  *


See also the :ref:`index of all functions and classes<genindex>`.

Log Levels
----------

MLflow Python APIs log information during execution using the Python Logging API. You can 
configure the log level for MLflow logs using the following code snippet. Learn more about Python
log levels at the
`Python language logging guide <https://docs.python.org/3/howto/logging.html>`_.

.. code-block:: python

    import logging

    logger = logging.getLogger("mlflow")

    # Set log level to debugging
    logger.setLevel(logging.DEBUG)

Known Limitations
-----------------

.. note::

   Plotly map figures that rely on external tile sources (e.g. ``mapbox_style``, ``carto-positron``,
   ``open-street-map``) may show a blank base map when viewed in the MLflow artifacts HTML viewer.
   The viewer runs in a sandboxed iframe, which blocks external tile server requests (CORS).
   Legends and choropleth colors still render correctly, but base tiles do not.

   **Workaround:** use a non-tile choropleth that renders fully offline:

   .. code-block:: python

      fig = px.choropleth(
          df,
          geojson=counties,
          locations="fips",
          color="unemp",
          scope="usa",
      )
      with mlflow.start_run():
          mlflow.log_figure(fig, "map.html")
