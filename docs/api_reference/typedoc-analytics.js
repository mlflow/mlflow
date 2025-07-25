// Google Analytics for TypeDoc generated pages
(function() {
  // Create the Google Analytics script tag
  var gtagScript = document.createElement('script');
  gtagScript.async = true;
  gtagScript.src = 'https://www.googletagmanager.com/gtag/js?id=AW-16857946923';

  // Insert it as the first script in the head
  var firstScript = document.getElementsByTagName('script')[0];
  if (firstScript && firstScript.parentNode) {
    firstScript.parentNode.insertBefore(gtagScript, firstScript);
  } else {
    document.head.appendChild(gtagScript);
  }

  // Initialize gtag
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'AW-16857946923');
})();