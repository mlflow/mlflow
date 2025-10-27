export function onRouteDidUpdate({ location }) {
  const { pathname } = location;

  document.body.classList.remove('mlflow-ml-section', 'mlflow-genai-section');

  if (pathname.startsWith('/genai')) {
    document.documentElement.setAttribute('data-genai-theme', 'true');
    document.body.classList.add('mlflow-genai-section');
  } else if (pathname.startsWith('/ml')) {
    document.documentElement.removeAttribute('data-genai-theme');
    document.body.classList.add('mlflow-ml-section');
  } else {
    document.documentElement.removeAttribute('data-genai-theme');
  }
}
