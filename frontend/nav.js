// Sets the active class on the nav link matching the current page filename.
(function () {
  const current = window.location.pathname.split('/').pop() || 'index.html';
  document.querySelectorAll('.nav-links a').forEach(function (link) {
    const href = link.getAttribute('href');
    if (href === current) {
      link.classList.add('active');
    }
  });
})();
