#### Example of interface to Python with Pyo3
 
 This is a small example to import a function from Rust into Python as a module. 
 You can pass numpy-arrays directly from Python to the Rust function and the function will also return numpy-arrays. 
 Compared to a Fortran module (compiled with f2py) the runtime was 4 times shorter. The implementation
 in Rust looks as follows for the calculation of Mulliken-charges
```rust
use ndarray::{ArrayViewD, Array1, ArrayView1, ArrayView2};
//use pyo3::prelude::{pymodule, pyfunction, PyModule, PyResult, Python};
use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArrayDyn, PyReadonlyArray1, PyReadonlyArray2};

#[pymodule]
fn rust_ext(_py: Python<'_>, m: &PyModule) -> PyResult<()> {

    // Mulliken Charges
    fn mulliken(p: ArrayView2<f64>, p0: ArrayView2<f64>,
                s: ArrayView2<f64>, orbs_per_atom: Vec<i64>,
                Nat: usize, Norb: u32)
                -> (Array1<f64>, Array1<f64>) {

        let dP = &p - &p0;

        let mut q = Array1::<f64>::zeros(Nat);
        let mut dq = Array1::<f64>::zeros(Nat);

        // iterate over atoms A
        let mut mu = 0;
        // WARNING: this loop cannot be parallelized easily because mu is incremented
        // inside the loop
        for A in 0..Nat {
            // iterate over orbitals on atom A
            for muA in 0..orbs_per_atom[A] {
                let mut nu = 0;
                // iterate over atoms B
                for B in 0..Nat {
                    // iterate over orbitals on atom B
                    for nuB in 0..orbs_per_atom[B] {
                        q[A] = q[A] + (&p[[mu,nu]] * &s[[mu,nu]]);
                        dq[A] = dq[A] + (&dP[[mu,nu]] * &s[[mu,nu]]);
                        nu += 1;
                    }
                }
                mu += 1;
            }
        }
        (q, dq)
    }

    // wrapper of mulliken
    #[pyfn(m, "mulliken")]
    fn mulliken_py<'py>(
        py: Python<'py>,
        p: PyReadonlyArray2<f64>,
        p0: PyReadonlyArray2<f64>,
        s: PyReadonlyArray2<f64>,
        orbs_per_atom: Vec<i64>,
        Nat: usize,
        Norb: u32,
    ) -> (&'py PyArray1<f64>, &'py PyArray1<f64>) {
        let p = p.as_array();
        let p0 = p0.as_array();
        let s = s.as_array();
        //let orbs_per_atom = orbs_per_atom.as_array();
        let ret = mulliken(p, p0, s, orbs_per_atom, Nat, Norb);
        (ret.0.into_pyarray(py), ret.1.into_pyarray(py))
    }

    Ok(())
}
```

The first function `fn rust_ext`  describes the Python module, which can then be imported. The function
mulliken' calculates the charges in Rust, but is not directly accessible for Python. Therefore it is necessary to
to wrap to make the file types compatible between Python and Rust. This is done in the function `fn mulliken_py`, 
the attribute `#[pyfn(m, "mulliken`)]` makes this function available as a Python function under the name 'mulliken'.
The necessary settings in Cargo.toml for this function were
```
[lib]
name = "rust_ext"
crate-type = ["cdylib"]

[dependencies]
numpy = "0.11.0"
ndarray = "0.13.0"

[dependencies.pyo3]
version = "0.11.1"
features = ["extension-module"]
```
and compiled the module with a setup.py. This had the following content:
````python
from setuptools import find_packages, setup
from setuptools_rust import RustExtension


setup_requires = ['setuptools-rust>=0.10.2']
install_requires = ['numpy']
test_requires = install_requires + ['pytest']

setup(
    name='rust_ext',
    version='0.1.0',
    description='Example of python-extension using rust-numpy',
    rust_extensions=[RustExtension(
        'rust_ext.rust_ext',
        './Cargo.toml',
        debug=False
    )],
    install_requires=install_requires,
    setup_requires=setup_requires,
    test_requires=test_requires,
    packages=find_packages(),
    zip_safe=False,
)
````
I tested the implementation with the following Python script. At first here are some DFTBaby calls
to calculate the necessary numpy-arrays as input: 
```python
import numpy as np
from DFTB.DFTB2 import DFTB2
from DFTB import XYZ

atomlist = XYZ.read_xyz("ethene_tetramer.xyz")[0]
atomlist = XYZ.read_xyz("opt.xyz")[0]
dftb = DFTB2(atomlist)
dftb.setGeometry(atomlist, charge=0)

dftb.getEnergy()

Nat = 12
Norb = 24

# here the import from the rust modul happens
from rust_ext import mulliken
q, dq = mulliken(dftb.p, dftb.p0, dftb.s, dftb.orbs_per_atom, Nat, Norb)
``` 