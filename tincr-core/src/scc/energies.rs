use ndarray::prelude::*;

// compute HOMO-LUMO gap in Hartree, should be moved to DATA
pub fn get_homo_lumo_gap(orbe: ArrayView1<f64>, homo_lumo_idx: (usize, usize)) -> f64 {
    orbe[homo_lumo_idx.1] - orbe[homo_lumo_idx.0]
}

pub fn lc_exchange_energy(
    s: ArrayView2<f64>,
    g0_lr_ao: ArrayView2<f64>,
    p0: ArrayView2<f64>,
    p: ArrayView2<f64>,
) -> f64 {
    let dp: Array2<f64> = &p - &p0;
    let mut e_hf_x: f64 = 0.0;
    e_hf_x += ((s.dot(&dp.dot(&s))) * &dp * &g0_lr_ao).sum();
    e_hf_x += (s.dot(&dp) * dp.dot(&s) * &g0_lr_ao).sum();
    e_hf_x *= -0.125;
    return e_hf_x;
}

/// Compute electronic energies
pub fn get_electronic_energy(
    p: ArrayView2<f64>,
    p0: ArrayView2<f64>,
    s: ArrayView2<f64>,
    h0: ArrayView2<f64>,
    dq: ArrayView1<f64>,
    gamma: ArrayView2<f64>,
    g0_lr_ao: Option<ArrayView2<f64>>,
) -> f64 {
    // band structure energy
    let e_band_structure: f64 = (&p * &h0).sum();

    // Coulomb energy from monopoles
    let e_coulomb: f64 = 0.5 * &dq.dot(&gamma.dot(&dq));

    // electronic energy as sum of band structure energy and Coulomb energy
    let mut e_elec: f64 = e_band_structure + e_coulomb;

    // add lc exchange to electronic energy if lrc is requested
    if g0_lr_ao.is_some() {
        let e_hf_x: f64 = lc_exchange_energy(s, g0_lr_ao.unwrap(), p0, p);
        e_elec += e_hf_x;
    }

    return e_elec;
}

pub fn get_electronic_energy_unrestricted(
    p_alpha: ArrayView2<f64>,
    p_beta: ArrayView2<f64>,
    h0: ArrayView2<f64>,
    dq_alpha: ArrayView1<f64>,
    dq_beta: ArrayView1<f64>,
    gamma: ArrayView2<f64>,
    spin_couplings:ArrayView1<f64>,
) -> f64 {
    let dq:Array1<f64> = &dq_alpha + &dq_beta;
    let m_squared: Array1<f64> = (&dq_alpha - &dq_beta).iter().map(|x| x * x).collect();

    // band structure energy
    let e_band_structure: f64 = (&(&p_alpha+&p_beta) * &h0).sum();

    // Coulomb energy from monopoles
    let e_coulomb: f64 = 0.5 * &dq.dot(&gamma.dot(&dq));

    // Spin polarization energy
    let e_spin: f64 = 0.5 * m_squared.dot(&spin_couplings);

    // electronic energy as sum of band structure energy and Coulomb energy
    let mut e_elec: f64 = e_band_structure + e_coulomb + e_spin;

    return e_elec;
}