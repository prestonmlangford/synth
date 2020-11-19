use rand::Rng;
use rand::distributions::Uniform;
use rayon::prelude::*;

#[derive(Clone)]
pub struct Solution {
    pub position: Vec<f64>,
    pub fitness: f64,
}

#[derive(Clone)]
pub struct DE {
    cr: f64,
    f: f64,
    np: usize,
    d: usize,
    max: f64,
    pop: Vec<Solution>,
    fit: fn(position: Vec<f64>) -> f64,
    ready: bool,
    best: usize,
}

fn choose(count: usize, lo: usize, hi: usize, avoid: usize) -> Vec<usize> {
    let mut selection = vec![avoid,1];
    let mut rng = rand::thread_rng();
    while selection.len() < count + 1 {
        let r = rng.gen_range(lo, hi);
        if !selection.contains(&r) {
            selection.push(r);
        }
    }
    
    selection[1..].to_vec()
}

impl DE {
    fn step(&mut self) -> Result<(),&'static str> {
        if !self.ready {
            return Err("Must call init before use!");
        }
        let urange = Uniform::new_inclusive(0.0, 1.0);
        
        self.pop = self.pop.par_iter().enumerate()
            .map(|(i,x)| {
                let mut rng = rand::thread_rng();
                let sel = choose(3, 0, self.pop.len(), i);
                let r = rng.gen_range(0, self.d);
                let mut y = x.clone();
                
                for i in 0..y.position.len() {
                    let u = rng.sample(urange);
                    if (i == r) || (u < self.cr) {
                        let a = self.pop[sel[0]].position[i];
                        let b = self.pop[sel[1]].position[i];
                        let c = self.pop[sel[2]].position[i];
                        y.position[i] = a + self.f*(b - c);
                    }
                }
                y.fitness = (self.fit)(y.position.clone());
            
                if y.fitness <= x.fitness {
                    y
                } else {
                    x.clone()
                }
            }).collect();
        
        let (best,_) = self.pop.iter().enumerate()
            .fold((0,0.0),|(b,m),(i,x)| if x.fitness <= m {(i,x.fitness)} else {(b,m)});
        
        self.best = best;
        
        return Ok(())
    }
    
    pub fn new(dimensions: usize, function: fn(Vec<f64>) -> f64) -> Self {
        DE {
            cr: 0.9,
            f: 0.8,
            np: 10*dimensions,
            max: 1.0,
            d: dimensions,
            pop: vec![],
            fit: function,
            ready: false,
            best: 0,
        }
    }
    
    pub fn crossover_probability(mut self,p: f64) -> Self {
        self.cr = p;
        self
    }
    
    pub fn differential_weight(mut self,w: f64) -> Self {
        self.f = w;
        self
    }
    
    pub fn population_size(mut self,n: usize) -> Self {
        self.np = n;
        self
    }
    
    pub fn bound(mut self,b: f64) -> Self {
        self.max = b;
        self
    }
    
    pub fn init(mut self) -> Self {
        let mut rng = rand::thread_rng();
        let urange = Uniform::new_inclusive(0.0, 1.0);
        for _ in 0..self.np {
            
            let position: Vec<f64> = (0..self.d).map(|_| 1.0 + self.max*(rng.sample(urange) - 0.5)).collect();
            let fitness = (self.fit)(position.clone()); 
            let solution = Solution{position, fitness};
            self.pop.push(solution);
        }
        self.ready = true;
        self
    }    
}

impl Iterator for DE {
    type Item = Solution;
    
    fn next(&mut self) -> Option<Solution> {
        match self.step() {
            Ok(_) => Some(self.pop[self.best].clone()),
            Err(e) => {
                println!("{:?}",e); 
                None
            }
        }
    }
}