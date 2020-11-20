const FS: f64 = 44100.0;

mod de;
mod ks;
mod wav;

#[macro_use]
extern crate lazy_static;


lazy_static! {
    static ref IDEAL: Vec<f64> = wav::read("ideal.wav");
}


fn corr(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    let mb: f64 = b.iter().sum::<f64>() / (b.len() as f64);
    let ma: f64 = a.iter().sum::<f64>() / (a.len() as f64);
    let va: f64 = a.iter().map(|x| {let d = x - ma; d*d}).sum::<f64>();
    let vb: f64 = b.iter().map(|x| {let d = x - mb; d*d}).sum::<f64>();

    a.iter().zip(b.iter()).
    map(|(ai,bi)| (ai - ma)*(bi - mb)).
    sum::<f64>()/(va*vb).sqrt()
}

fn fitness(h: Vec<f64>) -> f64 {
    let n = IDEAL.len();
    let f = 196.0;
    let test = ks::ks(h,f,n);
    corr(&test,&IDEAL)
}

fn make_pluck(h: &Vec<f64>){
    let y = ks::ks(h.clone(),196.0,(4.0*FS) as usize);
    wav::write(y, "pluck.wav");
}

fn main() {
    use std::fs::OpenOptions;
    use std::io::prelude::*;

    let mut file = OpenOptions::new()
        .truncate(true)
        .append(true)
        .create(true)
        .open("coefficients.log")
        .unwrap();

    let dimension = 16;
    let de = 
        de::DE::new(dimension,fitness).
        bound(0.1). //0.99 / (dimension as f64)
        init();
    
    for solution in de {
        println!("{:?}",solution.fitness);
        
        make_pluck(&solution.position);
        
        if let Err(e) = writeln!(file, "{:?}\n",solution.position) {
            eprintln!("Couldn't write to file: {}", e);
        }
    }
}