// var digits = 10; 
    // seed = 1234567893; 

// function nextRand() {
//     var n = (seed * seed).toString(); 
//     while(n.length < digits * 2)
//     {
//         n = "0" + n; 
//     }
//     var start = Math.floor(digits / 2), 
//         end = start + digits; 
//     seed = parseInt(n.substring(start, end)); 
//     return seed; 
// }

// function nextRandFloat() {
//     return nextRand() / 9999999999; 
// }

// for(var i = 0; i < 20; i++)
// {
//     console.log(nextRandFloat());
// }

// var results = []; 
// for (var i = 0; i < 100000; i++)
// {
//     var rand = nextRand();
//     if(results[rand]){
//         break;
//     } 
//     results[rand] = true; 
// }
// console.log(i); 


// linear congruential generator
var a = 1664525, 
    c = 1013904223, 
    m = Math.pow(2, 32), 
    seed = 958736; 

function nextRand() 
{
    seed = (a * seed + c) % m; 
    return seed; 
}

function nextRandFloat()
{
    return nextRand() / m; 
}


var canvas = document.getElementById("random"), 
    ctx = canvas.getContext("2d"); 

var y = 0; 
draw(); 

function draw()
{
    for (var x = 0; x < 60; x++)
    {
        if (nextRandFloat() < 0.5){
            ctx.fillRect(x, y, 1, 1); 
        }
    }
    y++; 
    requestAnimationFrame(draw);
}

for (var i = 0; i < 5; i++)
{
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    requestAnimationFrame(draw);
}
