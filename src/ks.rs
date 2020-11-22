use crate::FS;


struct Filter {
    cof: Vec<f64>,
    buf: Vec<f64>,
    idx: usize
}


impl Filter {
    fn next(&mut self, x: f64) -> f64 {
        if self.idx == 0 {
            self.idx = self.buf.len() - 1;
        } else {
            self.idx -= 1;
        }
        self.buf[self.idx] = x;
        
        let lo = (&self.buf[self.idx..]).iter();
        let hi = (&self.buf[..self.idx]).iter();
        let buf = lo.chain(hi);
        let cof = self.cof.iter();
        
        cof.zip(buf).
        map(|(x,y)| x * y).
        fold(0.0, |sum,x| sum + x)
    }


    fn new(coefficients: Vec<f64>) -> Filter {
        let len = coefficients.len();
        Filter {
            cof: coefficients,
            buf: vec![0.0;len],
            idx: 0
        }
    }

    fn lagrange_delay(n: usize, d: f64) -> Filter {
        let w = d.floor() as usize;
        let f = d - (w as f64);

        let even = (n & 1) == 0;
        let mid = if even {
                n/2
            } else {
                if f < 0.5 {
                    n/2 + 1
                } else {
                    n/2
                }
            };
        
        
        if w < mid {
            panic!("Lagrange delay must be at least half as long as the filter order")
        }
        let num_zeros = w - mid;
        let mut z: Vec<f64> = vec![0.0; num_zeros];

        let ld = d - (num_zeros as f64);
        let mut lag: Vec<f64> = vec![1.0; n + 1];

        for i in 0..(n+1) {
            for k in 0..(n+1) {
                if i != k {
                    let fi = i as f64;
                    let fk = k as f64;
                    lag[i] *= (ld - fk)/(fi - fk);
                }
            }
        }

        z.append(&mut lag);
        
        Filter::new(z)
    }
}

fn normalize(h: Vec<f64>) -> Vec<f64> {
    use rustfft::algorithm::Radix4;
    use rustfft::FFT;
    use rustfft::num_complex::Complex;
    
    let fft = Radix4::new(h.len(), false);
    let mut input: Vec<Complex<f64>> = h.iter().map(|x| Complex::new(*x, 0.0)).collect();
    let mut output: Vec<Complex<f64>> = input.clone();
    
    fft.process(&mut input[..], &mut output);
    
    let max = 
        output.iter().
        map(|c| c.norm()).
        fold(0.0, |max,x| if x > max {x} else {max});
    
    // println!("max gain {:?}",max);
    // for (i,c) in output.iter().map(|c| c.norm()).enumerate() {
    //    println!("output[{}] = {}",i,c);
    // }
    let eps = 0.00000001;
    h.iter().map(|x| 0.999*x/(max + eps)).collect()
}

pub fn ks(body: Vec<f64>,freq: f64,n: usize) -> Vec<f64> {
    use rand_distr::{Normal, Distribution};
    use rand::thread_rng;
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut randn = normal.sample_iter(thread_rng());
    
    let mut y: Vec<f64> = vec![0.0;n];
    
    let correction = (body.len()/2) as f64;
    let mut d = Filter::lagrange_delay(1, FS/freq - correction);
    
    let nbody = normalize(body);
    let mut h = Filter::new(nbody);
    
    let input_len = (FS/freq) as usize;
    
    let mut x = 1.0;
    for i in 0..n {
        // let x = if i < input_len {
        //     let r = randn.next().unwrap();
        //     //let r = 0.01 * ((i as f64) - ((ramp_len/2) as f64));
        //     r + fb
        // } else {
        //     fb
        // };
        
        y[i] = h.next(x);
        x = d.next(y[i]);
    }
    
    let max = 
        y.iter().
        map(|x| x.abs()).
        fold(0.0, |max,x| if x > max {x} else {max});
    
    let eps = 0.000000001;
    y.iter().map(|x| x/(max + eps)).collect()
}
