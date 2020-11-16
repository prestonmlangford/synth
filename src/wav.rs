use std::i16;

pub fn read(file: &str) -> Vec<f64> {
    let mut reader = hound::WavReader::open(file).unwrap();
    let data: Vec<f64> = reader.samples::<i16>().map(|s| s.unwrap() as f64).collect();
    
    let max = 
        data.iter().
        map(|x| x.abs()).
        fold(0.0, |max,x| if x > max { x } else { max });
        
    data.iter().map(|d| d/max).collect()
}

pub fn write(data: Vec<f64>, file: &str) {
    let max = 
        data.iter().
        map(|x| x.abs()).
        fold(0.0, |max,x| if x > max { x } else { max });
    
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 44100,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    
    let mut writer = hound::WavWriter::create(file, spec).unwrap();
    let amplitude = 0.25 * (i16::MAX as f64);
    
    for sample in data.iter() {
        let scaled = (sample * amplitude / max) as i16;
        writer.write_sample(scaled).unwrap();
    }
}