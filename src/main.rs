struct Filter {
    cof: Vec<f64>,
    buf: Vec<f64>,
    idx: usize
}


impl Filter {
    fn push(&mut self, x: f64) {
        self.buf[self.idx] = x;
        self.idx = (self.idx + 1) % self.buf.len();
    }

    fn calc(&self) -> f64 {
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

fn main() {
    println!("Hello, world!");

    let lag = Filter::lagrange_delay(10,44100.0/110.4);

    for c in lag.cof.iter() {
        print!("{:?} ",c);
    }
}