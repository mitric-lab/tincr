<div align="left">
  <img src="https://github.com/hochej/tincr/blob/master/tincr.png" height="120"/>
</div>

## Install
##### Warning: The features of the program are currently very limited! 
To be able to compile and run the __tincr__ program it is necessary to have the Intel MKL libary
installed, as the linear algebra operations are based on the MKL LAPACK and BLAS implementations. 
You can just download the MKL library from the Intel webpage and after installation make sure that the 
enviroment variables are set. If not you can just execute the `mklvars.sh` script which is located in the
installation directory of the MKL Library. 
```bash
source /path/to/MKL/mklvars.sh intel64
```  
Make sure that the environment variable `$MKLROOT` was set. 
Of course you also need Rust itself. This is straightforward to install and explained in 
detail on the [official site](https://www.rust-lang.org/tools/install). Furthermore, you need the [Rusty-FITPACK](https://github.com/mitric-lab/Rusty-FITPACK)([see Documentation for details](http://jhoche.de/Rusty-FITPACK/rusty_fitpack/) crate
for the spline interpolation. This can be cloned from the Github repository and installed in the same way.

Then just clone the repository to your local machine
```bash
git clone https://github.com/hochej/tincr.git
```
Go into the new directory
```bash
cd tincr
```
and build the executable with the package manager Cargo
```bash
cargo build --release
```
The option `--release` enables all optimization during the build and ensures fast runtimes, but can
result in very long compile times. If you want to compile often e.g. in the case of debugging, then 
it makes sense to just execute
```bash
cargo build
``` 
To be able to execute the `tincr` programm you should set `TINCR_SRC_DIR` to the installation directory and you 
can add the binary path to your `PATH` environment variable.

### Example
This example shows the installation on the wuxcs cluster as a local user: 
```bash
source /opt/local/intel/compilers_and_libraries_2019.4.243/linux/mkl/bin/mklvars.sh intel64
cd $HOME/software
git clone https://github.com/mitric-lab/Rusty-FITPACK.git
git clone https://github.com/mitric-lab/tincr
cd tincr
```
Update the path to the Rusty-Fitpack directory in `Cargo.toml`
```
cargo build --release
export TINCR_SRC_DIR="$HOME/software/tincr"
export PATH=$PATH:$TINCR_SRC_DIR/target/release
```


### Ideas for new quantum-chemical features
The idea of this project is to port the DFTBaby program package by Alexander Humeniuk 
in Rust and to expand it. The advantages would be that the efficiency can be significantly
improved and the programme would be easier to run in parallel.
Furthermore, fragment-orbital based calculations based on DFTB will 
be implemented (FMO method).

- **Archetype is DFTBaby**
- spin-unrestricted DFTB would improve description of triplets
- machine-learned repulsive potentials as done in Orb-Net
- https://aip.scitation.org/doi/pdf/10.1063/5.0020545
- FMO-method as based on monomer calculations
    - FMO-DFTB: https://pubs.acs.org/doi/10.1021/ct500489d
    - FMO-LC-DFTB: https://pubs.acs.org/doi/pdf/10.1021/acs.jctc.9b00108
- Spin-Orbit-Couplings: https://pubs.acs.org/doi/10.1021/acs.jctc.6b00915
    - Implementation https://github.com/jzpathfinder/pysoc
    

### Ideas for the technical development of the program
- in general functional programming style should be prefered over OOP 
    - pure functions and functions withoud side effects would be the best way, but sometimes create some boilercode

#### Input/Output-Format
##### Input
- Command line input in the case of single point and optimization
- in the case of all other calculations the Q-Chem input format could be a nice way

##### Ouput
- the normal output should be written with a logger 
The packet [log](https://docs.rs/log/0.4.11/log/) could be suitable for this.  
- the .traj format of ASE would be practical as output format
for optimisation/dynamics (https://gitlab.com/ase/ase/-/blob/master/ase/io/trajectory.py)
- additional quantum chemistry output could be written to a .fchk file. The advantage would be here,
that the format is easy to write and can already be read by many programs
- Orbital information can be written to .molden files as is already the case in DFTBabyt

#### Parameters
The description of how to use the parameters can be found on the [extra page](https://github.com/hochej/tincr/blob/master/Parameter.md)

#### Units and Constants
- the management of the units could be taken over from the ASE concept, see https://gitlab.com/ase/ase/-/blob/master/ase/units.py
- other constants could be defined as `const` or `static` with (static lifetime) 

#### Efficiency
- for working with arrays and linear algebra the [ndarray](https://docs.rs/ndarray/0.13.1/ndarray/) package seems to be well suited
- perhaps the [smallvec](https://crates.io/crates/smallvec) package should also be considered 

#### Memory management 
- there would have to be a check on how big the molecule is and whether the matrices still fit into the RAM,
otherwise the arrays must be written to the hard disk
- otherwise you could try to calculate many arrays not in the whole, but only the elements 
sequentially 

#### Interface (API) to Python
An interface to Python can be implemented quite easily with the [PyO3] module (https://github.com/PyO3/pyo3). A concrete example 
in combination with numpy can be found here: https://github.com/PyO3/rust-numpy/tree/master/examples/simple-extension <br>
An example of my own for such an interface to Python can be found here: https://github.com/hochej/tincr/blob/master/PyO3_Beispiel.md

#### Documentation
Some notes on the documentation of the code are given on this [extra page](https://github.com/hochej/tincr/blob/master/Documentation.md).
Comments that start with three slashes are automatically added to the documentation
is displayed as text. 

-----------------------------------
### Starting with Rust
Enclosed are a few useful pages for learning Rust (besides the official [book](https://doc.rust-lang.org/book/)):

1. [Amos Wegner's A half-hour to learn Rust](https://fasterthanli.me/articles/a-half-hour-to-learn-rust)
2. [Richard Anaya's Tour of Rust](https://tourofrust.com/00_en.html)
3. [Rustlings debugging Aufgaben](https://github.com/rust-lang/rustlings)
4. on [Excercism.io](https://exercism.io/tracks/rust.) there are many tasks on Rust (light, medium, heavy)  
5. [Tutorial](https://www.philippflenker.com/hecto/) to write your own vim-like text editor in Rust

Very detailed [overview](https://github.com/Dhghomon/easy_rust/blob/master/README.md ) on Rust (quasi a book on Github)


[Infos on Rust modules](http://www.sheshbabu.com/posts/rust-module-system/) <br>
[Infos on Lifetimes](https://github.com/pretzelhammer/rust-blog/blob/master/posts/common-rust-lifetime-misconceptions.md) <br> 
[Infos on Sizedness](https://github.com/pretzelhammer/rust-blog/blob/master/posts/sizedness-in-rust.md) <br>
 
All these suggestions are taken from the official [Rust Forum](https://users.rust-lang.org/t/best-way-to-learn-rust-programming/47522/3
).
#### Editor/IDE for Rust
The IDE [CLion](https://www.jetbrains.com/clion/) from Jetbrains offers very good support for Rust (through a plugin). Additionally you can use
the git versioning is very easy. The package is [free of charge] for university employees and students (https://www.jetbrains.com/de-de/community/education/#students).

#### Rustfmt and Clippy 
There are two tools in Rust that are very helpful in everyday life. 
The first one is `rustfmt`, this can be easily called by 
```bash
rustfmt some_file.rs
``` 
and it automatically formats the Rust file into a legible format following the
official style guide. 
The second tool, Clippy, which I discovered relatively late, goes one step further
and suggests code changes to make the code more idiomatic and also more efficient.
The tool is also included in the standard Rust installation and can be started with:
```bash
cargo clippy
```


#### Rust naming conventions  

<p>In general, Rust tends to use <code>CamelCase</code> for &quot;type-level&quot; constructs
(types and traits) and <code>snake_case</code> for &quot;value-level&quot; constructs. More
precisely:</p>

<table><thead>
<tr>
<th>Item</th>
<th>Convention</th>
</tr>
</thead><tbody>
<tr>
<td>Crates</td>
<td><code>snake_case</code> (but prefer single word)</td>
</tr>
<tr>
<td>Modules</td>
<td><code>snake_case</code></td>
</tr>
<tr>
<td>Types</td>
<td><code>CamelCase</code></td>
</tr>
<tr>
<td>Traits</td>
<td><code>CamelCase</code></td>
</tr>
<tr>
<td>Enum variants</td>
<td><code>CamelCase</code></td>
</tr>
<tr>
<td>Functions</td>
<td><code>snake_case</code></td>
</tr>
<tr>
<td>Methods</td>
<td><code>snake_case</code></td>
</tr>
<tr>
<td>General constructors</td>
<td><code>new</code> or <code>with_more_details</code></td>
</tr>
<tr>
<td>Conversion constructors</td>
<td><code>from_some_other_type</code></td>
</tr>
<tr>
<td>Local variables</td>
<td><code>snake_case</code></td>
</tr>
<tr>
<td>Static variables</td>
<td><code>SCREAMING_SNAKE_CASE</code></td>
</tr>
<tr>
<td>Constant variables</td>
<td><code>SCREAMING_SNAKE_CASE</code></td>
</tr>
<tr>
<td>Type parameters</td>
<td>concise <code>CamelCase</code>, usually single uppercase letter: <code>T</code></td>
</tr>
<tr>
<td>Lifetimes</td>
<td>short, lowercase: <code>&#39;a</code></td>
</tr>
</tbody></table>