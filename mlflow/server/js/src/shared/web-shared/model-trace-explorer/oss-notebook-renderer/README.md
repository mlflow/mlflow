# Jupter Notebook Trace UI Renderer

This directory contains a standalone notebook renderer that is built as a separate entry point from the main MLflow application.

## Architecture

The notebook renderer is configured as a separate webpack entry point that generates its own HTML file and JavaScript bundle, completely independent of the main MLflow application.

### Build Configuration

The webpack configuration in `craco.config.js` handles the dual-entry setup:

1. **Entry Points**:

   - `main`: The main MLflow application (`src/index.tsx`)
   - `ml-model-trace-renderer`: The notebook renderer (`src/shared/web-shared/model-trace-explorer/oss-notebook-renderer/index.ts`)

2. **Output Structure**:

   ```
   build/
   ├── index.html                           # Main app HTML (excludes notebook renderer)
   ├── static/js/main.[hash].js             # Main app bundle
   ├── static/css/main.[hash].css           # Main app styles
   └── lib/notebook-trace-renderer/
       ├── index.html                       # Notebook renderer HTML
       └── js/ml-model-trace-renderer.[hash].js  # Notebook renderer bundle
   ```

3. **Path Resolution**:
   - Main app uses relative paths: `static-files/static/js/...`
   - Notebook renderer uses absolute paths: `/static-files/lib/notebook-trace-renderer/js/...`
   - Dynamic chunks use absolute paths: `/static-files/static/...` (via `__webpack_public_path__`)

### Key Configuration Details

#### Separate Entry Configuration

```javascript
webpackConfig.entry = {
  main: webpackConfig.entry, // Preserve original entry as 'main'
  'ml-model-trace-renderer': path.resolve(
    __dirname,
    'src/shared/web-shared/model-trace-explorer/oss-notebook-renderer/index.ts',
  ),
};
```

#### Output Path Functions

```javascript
webpackConfig.output = {
  filename: (pathData) => {
    return pathData.chunk.name === 'ml-model-trace-renderer'
      ? 'lib/notebook-trace-renderer/js/[name].[contenthash].js'
      : 'static/js/[name].[contenthash:8].js';
  },
  // ... similar for chunkFilename
};
```

#### HTML Plugin Configuration

- **Main app**: Excludes notebook renderer chunks via `excludeChunks: ['ml-model-trace-renderer']`
- **Notebook renderer**: Includes only its own chunks via `chunks: ['ml-model-trace-renderer']`

#### Runtime Path Override

The notebook renderer sets `__webpack_public_path__ = '/static-files/'` at runtime to ensure dynamically loaded chunks use the correct absolute paths.

## Files

- `index.ts`: Entry point that sets webpack public path and bootstraps the renderer
- `bootstrap.tsx`: Main renderer component
- `index.html`: HTML template for the standalone renderer
- `index.css`: Styles for the renderer

## Usage

The notebook renderer is built automatically as part of the main build process:

```bash
yarn build
```

This generates both the main application and the standalone notebook renderer, accessible at:

- Main app: `/static-files/index.html`
- Notebook renderer: `/static-files/lib/notebook-trace-renderer/index.html`

## Development Notes

- The renderer is completely independent of the main app - no shared runtime dependencies
- Uses absolute paths to avoid complex relative path calculations
- Webpack code splitting works correctly for both entry points
- CSS extraction is configured separately for each entry point
