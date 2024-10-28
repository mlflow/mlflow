sketch = """
<html>
  <head>
    <script>
      var traceData = JSON.parse({traces});
      window.addEventListener('message', (event) => {{
        if (event.data.type === 'READY') {{
          document.getElementById('trace-renderer').contentWindow.postMessage({{ 
            type: 'UPDATE_TRACE', 
            traceData: traceData, 
          }}, "*");
        }}
      }});
    </script>
  </head>
  <div>
    <iframe id="trace-renderer" style="width: 100%; height: 500px;" src="http://localhost:3000/static-files/lib/ml-model-trace-renderer/index.html" />
  </div>
</html>
"""