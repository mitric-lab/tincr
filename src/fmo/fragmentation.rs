use crate::constants::{BOND_THRESHOLD, ATOM_NAMES};
use chemfiles::{Atom, Frame, Trajectory};
use std::collections::HashMap;

pub fn get_fragments(frame: Frame) -> Vec<Frame>{
    // coordinates that were already seen
    let mut positions_seen: Vec<[f64; 3]> = Vec::new();

    // atomic numbers
    let mut numbers_seen: Vec<u8> = Vec::new();

    // for each monomer the atomic number and coordinates are stored
    let mut molecules: HashMap<usize, Vec<(u8, [f64; 3])>> = HashMap::new();

    // conversion of atomic index to atomic numbers and positions
    let mut idxtomol: HashMap<usize, usize> = HashMap::new();

    // atomic numbers of the whole system
    let atomic_numbers: Vec<u8> = (0..frame.size() as usize)
        .map(|i| frame.atom(i).atomic_number() as u8)
        .collect();

    // true -> create new fragment, false -> append to fragment
    let mut iflag: bool;

    // i: idx, zi: atomic number, posi: xyz coordinates in angstrom
    for (i, (zi, posi)) in atomic_numbers
        .iter()
        .zip(frame.positions().iter())
        .enumerate()
    {
        iflag = true;
        // iterate over the atoms that we have seen already in reversed order
        // since the probability that the current atom is linked to the last one
        // is much higher than that is linked to the first one.
        // j: idx, zj: atomic_number, posj: xyz coordinates in angstrom
        'inner: for (j, (zj, posj)) in numbers_seen
            .iter()
            .zip(positions_seen.iter())
            .enumerate()
            .rev()
        {
            // distance between atom i and atom j in angstrom
            let r: f64 = posi
                .iter()
                .zip(posj.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f64>()
                .sqrt();
            // if the atom is linked to an atom that we have already seen, the atom will be added
            // to this molecule. otherwise a new molecule will be createad
            if r < BOND_THRESHOLD[*zi as usize][*zj as usize] {
                // append atomic number and coordinates
                molecules
                    .entry(idxtomol[&j])
                    .or_insert(Vec::new())
                    .push((*zi, *posi));
                // and idx to the molecule entry
                idxtomol.insert(i, idxtomol[&j]);
                iflag = false;
                break 'inner;
            }
        }
        numbers_seen.push(*zi);
        positions_seen.push(*posi);
        // if the atom is not connected than we create new list of numbers and coords
        if iflag {
            molecules.insert(i, vec![(*zi, *posi)]);
            idxtomol.insert(i, i);
        }
    }
    // create a [Frame](chemfiles::Frame) for each fragment
    let mut fragment_frames: Vec<Frame> = Vec::with_capacity(molecules.len());
    for (key, value) in molecules.into_iter() {
        let mut fragment: Frame = Frame::new();
        for (num, pos) in value.iter() {
            fragment.add_atom(&Atom::new(ATOM_NAMES[*num as usize]), *pos, None);
        }
        fragment_frames.push(fragment);
    }
    return fragment_frames;
}
