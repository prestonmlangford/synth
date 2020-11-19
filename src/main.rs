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
    
    let energy = 
        a.iter().zip(b.iter()).
        map(|(x,y)| {let s = x + y; s*s}).
        fold(0.0,|sum,x| sum + x);
    
    let squared_difference = 
        a.iter().zip(b.iter()).
        map(|(x,y)| {let d = x - y; d*d}).
        fold(0.0,|sum,x| sum + x);
        
    squared_difference/energy
}

fn fitness(h: Vec<f64>) -> f64 {
    let n = IDEAL.len();
    let f = 196.0;
    let test = ks::ks(h,f,n);
    mse(&test,&IDEAL)
}

fn make_pluck(h: Vec<f64>){
    let y = ks::ks(h,196.0,(4.0*FS) as usize);
    wav::write(y, "pluck.wav");
}

fn optimize(){
    
    let dimension = 16;
    let de = 
        de::DE::new(dimension,fitness).
        bound(0.1). //0.99 / (dimension as f64)
        init();
    
        
    for solution in de {
        println!("{:?} -> {:?}",solution.fitness,solution.position);
        make_pluck(solution.position);
    }
}

fn main() {
    //make_pluck();
    optimize();
}