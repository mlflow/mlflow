fetch('https://pypi.org/pypi/mlflow/json')
  .then((response) => response.json())
  .then((data) => {
    var versions = Object.keys(data.releases)
      // Drop dev/post/rc versions
      .filter(function (version) {
        return /^\d+(\.\d+){0,3}$/.test(version);
      })
      // Sort versions
      // https://stackoverflow.com/a/40201629
      .map((a) =>
        a
          .split('.')
          .map((n) => +n + 100000)
          .join('.'),
      )
      .sort()
      .map((a) =>
        a
          .split('.')
          .map((n) => +n - 100000)
          .join('.'),
      )
      .reverse();

    var latestVersion = versions[0];
    var docRegex = /\/docs\/(?<version>[^/]+)\//;
    var currentVersion = docRegex.exec(window.location.pathname).groups.version;
    var dropDown = document.createElement('select');
    dropDown.onchange = function () {
      var newHref = window.location.href.replace(docRegex, `/docs/${this.value}/`);
      window.location.href = newHref;
    };
    versions.forEach(function (version) {
      var option = document.createElement('option');
      option.value = version;
      option.selected = version === currentVersion;
      option.text = version === latestVersion ? `${version} (latest)` : version;
      dropDown.appendChild(option);
    });

    var versionTag = document.querySelector('span.version');
    versionTag.parentNode.replaceChild(dropDown, versionTag);
  })
  .catch((error) => {
    console.error('Failed to fetch package metadata from PyPI:', error);
  });
