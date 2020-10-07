### Beispiel für Interface zu Python mit Pyo3

Dies ist ein eigenes kleines Beispiel um eine Funktion aus Rust in Python als Modul zu importieren. 
Der Rust-Funktion kann man aus Python direkt numpy-arrays übergeben und die Funktion gibt auch numpy-arrays zürück. 
Die Laufzeit war im Vergleich zu einem Fortran-Modul (compiliert mit f2py) um einen Faktor 4 kürzer. Die Implementierung
in Rust sieht folgendermaßen aus für die Berechnung von Mulliken-Ladungen:
```rust
use ndarray::{ArrayViewD, Array1, ArrayView1, ArrayView2};
//use pyo3::prelude::{pymodule, pyfunction, PyModule, PyResult, Python};
use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArrayDyn, PyReadonlyArray1, PyReadonlyArray2};

#[pymodule]
fn rust_ext(_py: Python<'_>, m: &PyModule) -> PyResult<()> {

    // Mulliken Charges
    fn mulliken(P: ArrayView2<f64>, P0: ArrayView2<f64>,
                S: ArrayView2<f64>, orbsPerAtom: Vec<i64>,
                Nat: usize, Norb: u32)
                -> (Array1<f64>, Array1<f64>) {

        let dP = &P - &P0;

        let mut q = Array1::<f64>::zeros(Nat);
        let mut dq = Array1::<f64>::zeros(Nat);

        // iterate over atoms A
        let mut mu = 0;
        // WARNING: this loop cannot be parallelized easily because mu is incremented
        // inside the loop
        for A in 0..Nat {
            // iterate over orbitals on atom A
            for muA in 0..orbsPerAtom[A] {
                let mut nu = 0;
                // iterate over atoms B
                for B in 0..Nat {
                    // iterate over orbitals on atom B
                    for nuB in 0..orbsPerAtom[B] {
                        q[A] = q[A] + (&P[[mu,nu]] * &S[[mu,nu]]);
                        dq[A] = dq[A] + (&dP[[mu,nu]] * &S[[mu,nu]]);
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
        P: PyReadonlyArray2<f64>,
        P0: PyReadonlyArray2<f64>,
        S: PyReadonlyArray2<f64>,
        orbsPerAtom: Vec<i64>,
        Nat: usize,
        Norb: u32,
    ) -> (&'py PyArray1<f64>, &'py PyArray1<f64>) {
        let P = P.as_array();
        let P0 = P0.as_array();
        let S = S.as_array();
        //let orbsPerAtom = orbsPerAtom.as_array();
        let ret = mulliken(P, P0, S, orbsPerAtom, Nat, Norb);
        (ret.0.into_pyarray(py), ret.1.into_pyarray(py))
    }

    Ok(())
}
```

Die erste Funktion `fn rust_ext` beschreibt das Python-Modul, das anschließend importiert werden kann. Die Funktion
'mulliken' berechnet die Ladungen in Rust, ist aber für Python nicht direkt zugänglich. Daher ist es notwendig, diese
zu wrappen, um die Dateitypen zwischen Python und Rust kompatibel zu machen. Dies geschieht in der Funktion `fn mulliken_py`, 
durch das Attribut `#[pyfn(m, "mulliken`)]` wird diese Funktion als Python-Funktion verfügbar gemacht unter dem Namen 'mulliken'.
Die notwendigen Einstellungen in der Cargo.toml für diese Funktion waren:
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
und kompiliert wurde das Modul mit einer setup.py. Diese hatte den folgenden Inhalt:
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
Die Implementierung hatte ich dann mit folgendem Python-Skript getestet. Zunächst kommen hier einige DFTBaby-Aufrufe
um die notwendigen numpy-arrays als Input zu berechnen: 
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
q, dq = mulliken(dftb.P, dftb.P0, dftb.S, dftb.orbsPerAtom, Nat, Norb)
``` 