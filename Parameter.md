## Handling the DFTB parameters

This section concerns the following parameters: 
 - confined pseudo atoms
 - free pseudo atoms 
 - pairwise Slater-Koster tables
 - pairwise Repulsive-Potential tables
    
In DFTBaby the "homegrown" parameters are stored as Python files (module) for 
and there is the possibility to download the parameter files from
DFTB+ or Hotbit. Since it makes little sense in Rust to read Python files 
to store data, another solution must be found here. I had
first of all also considers to write all parameters as Rust files and
at compile time and "bake" them into the binary. However
to call up the data is then somewhat cumbersome.  There are also two
very useful articles on Github about saving data in Rust: 
[Global Data in Rust](https://github.com/paulkernfeld/global-data-in-rust)
and [Contiguous Data in Rust](https://github.com/paulkernfeld/contiguous-data-in-rust)

Currently, I think it makes most sense to load the parameters at the beginning of the runtime,
as the data volumes are not very large and this should be done efficiently. The package
[serde](https://serde.rs) seems to be very well suited for this. This allows efficient data 
load 'structs' from various file formats into Rust. 
I had initially worked with JSON files, but they have some very annoying disadvantages. 
In JSON there are very few data types (numbers, objects, arrays, strings), so it is for example
not possible to use a tuple as key for a dictionary/map. Therefore this can only be done by 
bypass the use of strings and this makes deserialisation much more complex. 
Therefore I think [Rusty Object Notation (RON)](https://github.com/ron-rs/ron) is the more suitable file format. 
This is similar to JSON, but contains exactly the needed features. 
As an introduction for the use of serde in combination with JSON files I found this [Video](https://www.youtube.com/watch?v=hIi_UlyIPMg) helpful.
Infos about serde's default declaration, which I often use, can be found here: [link](https://serde.rs/field-attrs.html)  

The DFTB+ and Hotbit parameters would then either have to be read in manually or converted
to RON files to use uniform parameter file formats. 

<s> The spline interpolation of the potentials should probably be best done with the
Package Peroxides can be made </s>. In DFTBaby the fitting of the cubic splines
is done with SciPy. The implementation in SciPy is a wrapper of the Fortran77
Package FITPACK from Paul Dierckxx. <s> Since I think that it is not worth the effort 
to implement the fitting of splines my self, it would be easiest to use the peroxides package. </s> Since the spline 
interpolation in peroxides did not really convince me, I implemented the splines myself. 
However, they are not part of tincr, but an external package (crate) and available at 
[rusty-FITPACK](https://github.com/mitric-lab/Rusty-FITPACK)

### Free and Confined Pseudo Atoms
<table>
<tr><th> Python (DFTBaby) </th><th> Rust (new) </th></tr>
<tr><td>

| Attribute            | Type             |
|----------------------|------------------|
| Z                    | int              |
| Nelec                | int              |
| r0                   | float            |
| r                    | ndarray          |
| radial_density       | ndarray          |
| occupation           | list[tuple(int, int, int)] |
| effective_potential  | ndarray          |
| orbital_names        | list[str]        |
| energies             | list[float]      |
| radial_wavefunctions | list[ndarray]    |
| angular_momenta      | list[int]        |
| valence_orbitals     | list[int]        |
| energy_1s            | float            |
| orbital_1s           | ndarray          |

</td><td>

| Attribute            | Type              |
|----------------------|-------------------|
| z                    | u8               |
| n_elec               | u8               |
| r0                   | f64             |
| r                    | Vec[f64]       |
| radial_density       | Vec[f64]       |
| occupation           | Vec[array[f64]]  |
| effective_potential  | Vec[f64]       |
| orbital_names        | Vec[str]         |
| energies             | Vec[f64]       |
| radial_wavefunctions | Vec[Vec[f64]] |
| angular_momenta      | Vec[u8]         |
| valence_orbitals     | Vec[u8]         |
| energy_1s            | f64             |
| orbital_1s           | Vec[f64]       |
</td><td>

</td></tr> </table>

### Slater-Koster tables


<table>
<tr><th> Python (DFTBaby) </th><th> Rust (new) </th></tr>
<tr><td>

| Attribute    | Type                        |
|--------------|-----------------------------|
| Dipole       | dict{tuple: ndarray[float]} |
| H            | dict{tuple: ndarray[float]} |
| s            | dict{tuple: ndarray[float]} |
| Z1           | int                         |
| Z2           | int                         |
| d            | ndarray[float]              |
| index2symbol | dict{int: str}              |

</td><td>

| Attribute       | Type             |
|-----------------|------------------|
| dipole          | HashMap<(u8, u8, u8), Vec<f64>> |
| h               | HashMap<(u8, u8, u8), Vec<f64>> |
| s               | HashMap<(u8, u8, u8), Vec<f64>> |
| z1              | u8               |
| z2              | u8               |
| d               | Vec<f64>         |
| index_to_symbol | HashMap<u8, String>  |

</td></tr> </table>


### Repulsive Pair Potential

<table>
<tr><th> Python (DFTBaby) </th><th> Rust (new) </th></tr>
<tr><td>


| Attribute   | Type          |
|-------------|---------------|
| Vrep        | ndarray[float]|
| Z1          | int           |
| Z2          | int           |
| d           | ndarray[float]|

</td><td>

| Attribute   | Type                |
|-------------|---------------------|
| vrep        | Vec<f64>                |
| z1          | u8                 |
| z2          | u8                 |
| d           | Vec<f64>           |

</td></tr> </table>
