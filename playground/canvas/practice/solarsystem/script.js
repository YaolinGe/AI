const sun = new Image();
const moon = new Image();
const earth = new Image();
const universe = new Image();
const canvas = document.getElementById("canvas");
// canvas.style.border = "2px solid red";
const ctx = canvas.getContext("2d");



function init() {
    sun.src = "sun.png";
    moon.src = "moon.png";
    earth.src = "earth.png";    
    universe.src = "universe.png";
    window.requestAnimationFrame(draw);
}

function draw() {
  ctx.globalCompositeOperation = "destination-over";
  ctx.clearRect(0, 0, 600, 600); // clear canvas

  ctx.fillStyle = "rgb(0 0 0 / 40%)";
  ctx.strokeStyle = "rgb(0 153 255 / 40%)";
  ctx.save();

  ctx.translate(300, 300);

  // Earth
  const time = new Date();
  ctx.rotate(
    ((2 * Math.PI) / 60) * time.getSeconds() + 
      ((2 * Math.PI) / 60000) * time.getMilliseconds(),
  );
  ctx.translate(160, 0);
  ctx.fillRect(0, -12, 500, 24); // Shadow
//   ctx.drawImage(earth, -12, -12);
  ctx.drawImage(earth, -12, -12, 24, 24);

  // Moon
  ctx.save();
  ctx.rotate(
    ((2 * Math.PI) / 6) * time.getSeconds() + 
      ((2 * Math.PI) / 6000) * time.getMilliseconds(),
  );
  ctx.translate(0, 28.5);
  ctx.drawImage(moon, -3.5, -3.5, 7, 7);
  ctx.restore();

  ctx.restore();

  ctx.beginPath();
  ctx.arc(300, 300, 160, 0, Math.PI * 2, false); // Earth orbit
  ctx.stroke();

  ctx.drawImage(sun, 250, 250, 100, 100);
  ctx.drawImage(universe, 0, 0, 600, 600);

  window.requestAnimationFrame(draw);
}

init();
