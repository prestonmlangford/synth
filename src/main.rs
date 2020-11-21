const FS: f64 = 44100.0;

mod de;
mod ks;
mod wav;

#[macro_use]
extern crate lazy_static;

lazy_static! {
    static ref IDEAL: Vec<f64> = wav::read("ideal.wav");
}

fn corr_slow(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    let mb: f64 = b.iter().sum::<f64>() / (b.len() as f64);
    let ma: f64 = a.iter().sum::<f64>() / (a.len() as f64);
    let va: f64 = a.iter().map(|x| {let d = x - ma; d*d}).sum::<f64>();
    let vb: f64 = b.iter().map(|x| {let d = x - mb; d*d}).sum::<f64>();

    a.iter().zip(b.iter()).
    map(|(ai,bi)| (ai - ma)*(bi - mb)).
    sum::<f64>()/(va*vb).sqrt()
}

fn corr(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    let n = a.len() as f64;
    let (sa,sb,sa2,sb2,sab) = a.iter().zip(b.iter()).
    fold((0.0,0.0,0.0,0.0,0.0), |(sa,sb,sa2,sb2,sab),(ai,bi)| (sa + ai,sb + bi,sa2 + ai*ai,sb2 + bi*bi,sab + ai*bi));

    (n*sab - sa*sb)/((n*sa2 - sa*sa)*(n*sb2 - sb*sb)).sqrt()
}

fn msenorm(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    let n = a.len() as f64;
    let (sd2,sa2,sb2) = a.iter().zip(b.iter()).
    fold((0.0,0.0,0.0), |(sd2,sa2,sb2),(ai,bi)| ({let d = ai - bi; sd2 + d*d},sa2 + ai*ai,sb2 + bi*bi));

    sd2/(sa2 + sb2)
}

fn fitness(h: Vec<f64>) -> f64 {
    let n = IDEAL.len();
    let f = 196.0;
    let test = ks::ks(h,f,n);
    -corr(&test,&IDEAL)
}

fn make_pluck(h: &Vec<f64>){
    let y = ks::ks(h.clone(),196.0,(4.0*FS) as usize);
    wav::write(y, "pluck.wav");
}

fn main() {
    let dimension = 256;
    let de = 
        de::DE::new(dimension,fitness).
        bound(0.1). //0.99 / (dimension as f64)
        population_size(500).
        init();
    
    for solution in de {
        println!("{:?}\n fitness: {:?}\n\n",solution.position,1.0-solution.fitness.abs());
        make_pluck(&solution.position);
    }
}
