const FS: f64 = 44100.0;

mod de;
mod ks;
mod wav;

#[macro_use]
extern crate lazy_static;

use std::time::Instant;
use rustfft::algorithm::Radix4;
use rustfft::FFT;
use rustfft::num_complex::Complex;

lazy_static! {
    static ref IDEAL: Vec<f64> = {
        let mut y = wav::read("ideal.wav");
        let n = y.len().next_power_of_two();
        let d = n - y.len();
        let mut z = vec![0.0;d];
        y.append(&mut z);
        y
    };
}

fn corr(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    
    let (sa,sb,sa2,sb2,sab) = 
        a.iter().zip(b.iter())
        .fold(
            (0.0,0.0,0.0,0.0,0.0), 
            |(sa,sb,sa2,sb2,sab),(ai,bi)| 
                (sa + ai,sb + bi,sa2 + ai*ai,sb2 + bi*bi,sab + ai*bi)
        );
    
    let n = a.len() as f64;
        
    (n*sab - sa*sb)/((n*sa2 - sa*sa)*(n*sb2 - sb*sb)).sqrt()
}

fn xcorr(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    
    let n = a.len();
    let fft = Radix4::new(n, false);
    let ifft = Radix4::new(n, false);
    let mut ac: Vec<Complex<f64>> = a.iter().map(|x| Complex::new(*x, 0.0)).collect();
    let mut bc: Vec<Complex<f64>> = b.iter().map(|x| Complex::new(*x, 0.0)).collect();
    let mut A: Vec<Complex<f64>> = vec![Complex::new(0.0,0.0);n];
    let mut B: Vec<Complex<f64>> = vec![Complex::new(0.0,0.0);n];
    
    fft.process(&mut ac[..], &mut A);
    fft.process(&mut bc[..], &mut B);
    
    B = B.iter().map(|c| c.conj()).collect();
    
    let mut AB: Vec<Complex<f64>> = A.iter().zip(B.iter()).map(|(x,y)| x*y).collect();
    let mut cc: Vec<Complex<f64>> = vec![Complex::new(0.0,0.0);n];
    
    ifft.process(&mut AB[..], &mut cc);
    
    let max = cc.iter().map(|c| c.norm()).fold(0.0,|m,x| if x > m {x} else {m});
    let anorm = a.iter().map(|x| x*x).sum::<f64>().sqrt();
    let bnorm = b.iter().map(|x| x*x).sum::<f64>().sqrt();
    
    max/(anorm*bnorm*(n as f64))
}

fn fitness(h: Vec<f64>) -> f64 {
    let n = IDEAL.len();
    let f = 196.0;
    let test = ks::ks(h.clone(),f,n);
    //let s = (h.iter().map(|x| x*x).sum::<f64>()/(h.len() as f64)).sqrt();
    //0.01*s + 1.0 - xcorr(&test,&IDEAL)
    1.0 - xcorr(&test,&IDEAL)
}

fn ackley(x: Vec<f64>) -> f64 {
    let a = 20.0;
    let b = 0.2;
    let c = 2.0*3.1415926535;
    let d = x.len() as f64;
    let s = (x.iter().map(|xi| xi*xi).sum::<f64>()/d).sqrt();
    let f = x.iter().map(|xi| (c*xi).cos()).sum::<f64>()/d;
    
    -a*(-b*s).exp() - f.exp() + a + (1.0 as f64).exp()
}

fn fitness2(h: Vec<f64>) -> f64 {
    h.iter().map(|x| x*x).sum::<f64>()/(h.len() as f64)
}

fn make_pluck(h: &Vec<f64>){
    let y = ks::ks(h.clone(),196.0,(4.0*FS) as usize);
    wav::write(y, "pluck.wav");
}

fn main() {
    let dimension = 64;
    let de = 
        de::DE::new(dimension,ackley).
        bound(10.0). //0.99 / (dimension as f64)
        population_size(10*dimension).
        init();

    let mut now = Instant::now();
    for (i,solution) in de.enumerate() {
        println!("{:?}\n fitness: {:?}, ms: {}, iteration: {}\n\n",
            solution.position,
            solution.fitness,
            now.elapsed().as_millis(),
            i
        );

        make_pluck(&solution.position);
        now = Instant::now();
    }
}
