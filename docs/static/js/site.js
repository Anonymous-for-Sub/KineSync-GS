(() => {
  const button = document.querySelector('.to-top');
  const update = () => button.classList.toggle('visible', window.scrollY > 700);
  window.addEventListener('scroll', update, { passive: true });
  button.addEventListener('click', () => window.scrollTo({ top: 0, behavior: 'smooth' }));

  document.querySelectorAll('video').forEach((video) => {
    video.addEventListener('play', () => {
      document.querySelectorAll('video').forEach((other) => {
        if (other !== video) other.pause();
      });
    });
  });
  update();
})();
