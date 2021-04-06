use ndarray::{s, Array, Array1, Array2, ArrayView2};

const CN_DEFAULT: f64 = 30.0; // Coordination number cutoff
const DISP2_DEFAULT: f64 = 60.0; // Two-body interaction cutoff
const DISP3_DEFAULT: f64 = 40.0; // Three-body interaction cutoff

// Usage: let rsc = RealspaceCutoffBuilder::new().[set_xx].build();
pub struct RealspaceCutoff {
    cn: f64,
    disp2: f64,
    disp3: f64,
}


pub struct RealspaceCutoffBuilder {
    cn: f64,
    disp2: f64,
    disp3: f64,
}

impl RealspaceCutoffBuilder {
    pub fn new() -> RealspaceCutoffBuilder {
        RealspaceCutoffBuilder {
            cn: CN_DEFAULT,
            disp2: DISP2_DEFAULT,
            disp3: DISP3_DEFAULT,
        }
    }

    pub fn set_cn(&mut self, cn: f64) -> &mut Self {
        self.cn = cn;
        self
    }

    pub fn set_disp2(&mut self, disp2: f64) -> &mut Self {
        self.disp2 = disp2;
        self
    }

    pub fn set_disp3(&mut self, disp3: f64) -> &mut Self {
        self.disp3 = disp3;
        self
    }

    pub fn build(&self) -> RealspaceCutoff {
        RealspaceCutoff {
            cn: self.cn,
            disp2: self.disp2,
            disp3: self.disp3,
        }
    }
}

/// Generate lattice points from repetitions
pub fn get_lattice_points_rep_3d(
    lat: ArrayView2<f64>,
    rep: &[usize; 3],
    origin: bool,
) -> Array2<f64> {
    let mut itr: usize = 0;
    if origin {
        let mut trans: Array2<f64> = Array::zeros(
            (rep.iter().map(|x| 2*x + 1).product::<usize>(), 3)
        );
        for ix in 0..rep[0] {
            for iy in 0..rep[1] {
                for iz in 0..rep[2] {
                    for jx in (mergei(-1, 1, ix > 0)..1+1).step_by(2).rev() {
                        for jy in (mergei(-1, 1, iy > 0)..1+1).step_by(2).rev() {
                            for jz in (mergei(-1, 1, iz > 0)..1+1).step_by(2).rev() {
                                trans.slice_mut(s![itr, ..]).assign(
                                    &(&lat.slice(s![0, ..]) * (ix as f64) * (jx as f64)
                                        + &lat.slice(s![1, ..]) * (iy as f64) * (jy as f64)
                                        + &lat.slice(s![2, ..]) * (iz as f64) * (jz as f64)));
                                itr += 1;
                            }
                        }
                    }
                }
            }
        }
        trans
    } else {
        let mut trans: Array2<f64> = Array::zeros(
            (rep.iter().map(|x| 2*x + 1).product::<usize>() - 1, 3)
        );
        for ix in 0..rep[0] {
            for iy in 0..rep[1] {
                for iz in 0..rep[2] {
                    if ix == 0 && iy == 0 && iz == 0 {
                        continue;
                    }
                    for jx in (mergei(-1, 1, ix > 0)..1+1).step_by(2).rev() {
                        for jy in (mergei(-1, 1, iy > 0)..1+1).step_by(2).rev() {
                            for jz in (mergei(-1, 1, iz > 0)..1+1).step_by(2).rev() {
                                trans.slice_mut(s![itr, ..]).assign(
                                    &(&lat.slice(s![0, ..]) * (ix as f64) * (jx as f64)
                                        + &lat.slice(s![1, ..]) * (iy as f64) * (jy as f64)
                                        + &lat.slice(s![2, ..]) * (iz as f64) * (jz as f64)));
                                itr += 1;
                            }
                        }
                    }
                }
            }
        }
        trans
    }
}

/// Create lattice points within a given cutoff
/// returns variable trans
fn get_lattice_points_cutoff(
    periodic: &[bool; 3],
    lat: &[[f64; 3]; 3],
    rthr: f64,
    ) -> Array2<f64> {
    if periodic.iter().any(|x| *x) {
        Array::zeros((1, 3))
    } else {
        let rep = get_translations(lat, rthr);
        let lat_a: Array2<f64> = lat_to_array(lat);
        get_lattice_points_rep_3d(lat_a.view(), &rep, true)
    }
}

/// Generate a supercell based on a realspace cutoff, this subroutine
/// doesn't know anything about the convergence behaviour of the
/// associated property.
fn get_translations(
    lat: &[[f64; 3]; 3],
    rthr: f64,
) -> [usize; 3] {
    // find normal to the plane...
    let normx: [f64; 3] = crossproduct(&lat[1], &lat[2]);
    let normy: [f64; 3] = crossproduct(&lat[2], &lat[0]);
    let normz: [f64; 3] = crossproduct(&lat[0], &lat[1]);
    // ...normalize it...
    let normx: [f64; 3] = new_3d_vec_from(normx.iter().map(|x| x/norm2(&normx)));
    let normy: [f64; 3] = new_3d_vec_from(normy.iter().map(|x| x/norm2(&normy)));
    let normz: [f64; 3] = new_3d_vec_from(normz.iter().map(|x| x/norm2(&normz)));
    // cos angles between normals and lattice vectors
    let cos10 = dot(&normx, &lat[0]);
    let cos21 = dot(&normy, &lat[1]);
    let cos32 = dot(&normz, &lat[3]);

    let rep: [usize; 3] = [
        (rthr/cos10).abs().ceil() as usize, // rep[0]
        (rthr/cos21).abs().ceil() as usize, // rep[1]
        (rthr/cos32).abs().ceil() as usize, // rep[2]
    ];

    return rep;
}

/// Returns tsource if mask is true, elsewise fsource.
/// Both sources should be isize.
fn mergei(tsource: isize, fsource: isize, mask: bool) -> isize {
    if mask {
        tsource
    } else {
        fsource
    }
}

/// Perform cross product of two 3D vectors.
fn crossproduct(
    a: &[f64; 3],
    b: &[f64; 3]
) -> [f64; 3] {
    let mut c: [f64; 3] = [0.0; 3];
    c[0] = a[1]*b[2] - b[1]*a[2];
    c[1] = a[2]*b[0] - b[2]*a[0];
    c[2] = a[0]*b[1] - b[0]*a[1];
    return c;
}

fn norm2(a: &[f64]) -> f64 {
    a.iter().map(|a| a.powi(2)).sum::<f64>().sqrt()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(a, b)| a * b).sum()
}

fn new_3d_vec_from<F: Iterator<Item=f64>>(src: F) -> [f64; 3] {
    let mut result = [0.0; 3];
    for (rref, val) in result.iter_mut().zip(src) {
        *rref = val;
    }
    result
}

fn lat_to_array(
    lat: &[[f64; 3]; 3],
) -> Array2<f64> {
    let mut lat_a: Array2<f64> = Array::zeros((3, 3));
    for i in 0..3 {
        lat_a.slice_mut(s![i, ..]).assign(&Array::from((&lat[i]).to_vec()));
    }

    return lat_a;
}