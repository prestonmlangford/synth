const FS: f64 = 44100.0;

mod de;
mod ks;
mod wav;

#[macro_use]
extern crate lazy_static;


lazy_static! {
    static ref IDEAL: Vec<f64> = wav::read("ideal.wav");
}


fn mse(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    let ita = a.iter();
    let itb = b.iter();
    
    ita.zip(itb).
    map(|(x,y)| {let d = x - y; d*d}).
    fold(0.0,|sum,x| sum + x)
}

fn fitness(h: Vec<f64>) -> f64 {
    let n = IDEAL.len();
    let f = 196.0;
    let test = ks::ks(h,f,n);
    mse(&test,&IDEAL)
}

fn main() {
    use std::process;

    
    
    let dimension = 10;
    let de = 
        de::DE::new(dimension,fitness).
        bound(0.99 / (dimension as f64)).
        init();
    
    // ctrlc::set_handler(|| {
    //     let best = de.best_solution();
    //     println!("");
    //     println!("{:?}",best.fitness);
    //     println!("{:?}",best.position);
    //     process::exit(0);
    // }).expect("Error setting Ctrl-C handler");
        
    for solution in de {
        
        println!("{:?} -> {:?}",solution.fitness,solution.position);
        
        
    }
}