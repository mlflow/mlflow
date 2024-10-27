sketch = """
<html>
  <head>
    <script>
      const traceData = {traces};
      const traceRenderer = document.getElementById('trace-renderer');
      window.addEventListener('message', (event) => {
        if (event.data.type === 'READY') {
          traceRenderer.contentWindow.postMessage({ 
            type: 'UPDATE_TRACE', 
            trace: traceData, 
          });
        }
      });
    </script>
  </head>
  <div>
    <iframe id="trace-renderer" src="http://localhost:3000/static-files/lib/ml-model-trace-renderer/index.html" />
  </div>
</html>
"""