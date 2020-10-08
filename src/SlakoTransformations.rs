/// compute directional cosines for the vector going from
/// pos1 to pos2
/// Returns:
/// ========
/// r: length of vector
/// x,y,z: directional cosines
fn directional_cosines(pos1: &[f64; 3], pos2: &[f64; 3]) -> (f64, f64, f64, f64) {
    let xc: f64 = pos2[0] - pos1[0];
    let yc: f64 = pos2[1] - pos1[1];
    let zc: f64 = pos2[2] - pos1[2];
    let r: f64 = (xc.powi(2) + yc.powi(2) + zc.powi(2)).sqrt();
    // directional cosines
    let x: f64;
    let y: f64;
    let z: f64;
    if r > 0.0 {
        x = xc / r;
        y = yc / r;
        z = zc / r;
    } else {
        x = 0.0;
        y = 0.0;
        z = 1.0;
    }
    return (r, x, y, z);
}


