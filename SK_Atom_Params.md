### Free and Confined Pseudo Atoms
<table>
<tr><th> Python (DFTBaby) </th><th> JSON (new) </th><th> Rust (new) </th></tr>
<tr><td>

| Attribute            | Type             |
|----------------------|------------------|
| Z                    | int              |
| Nelec                | int              |
| r0                   | float            |
| r                    | ndarray          |
| radial_density       | ndarray          |
| occupation           | list[tuple[int]] |
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
| z                    | number               |
| n_elec               | number               |
| r0                   | number             |
| r                    | array[number]       |
| radial_density       | array[number]       |
| occupation           | array[array[number]]  |
| effective_potential  | array[number]       |
| orbital_names        | array[string]         |
| energies             | array[number]       |
| radial_wavefunctions | array[array[number]] |
| angular_momenta      | array[number]         |
| valence_orbitals     | array[number]         |
| energy_1s            | number             |
| orbital_1s           | array[number]       |
</td><td>

| Attribute            | Type              |
|----------------------|-------------------|
| z                    | u8               |
| n_elec               | u8               |
| r0                   | f64             |
| r                    | array[f64]       |
| radial_density       | array[f64]       |
| occupation           | array[array[f64]]  |
| effective_potential  | array[f64]       |
| orbital_names        | array[str]         |
| energies             | array[f64]       |
| radial_wavefunctions | array[array[f64]] |
| angular_momenta      | array[u8]         |
| valence_orbitals     | array[u8]         |
| energy_1s            | number             |
| orbital_1s           | array[f64]       |
</td><td>

</td></tr> </table>

### Slater-Koster tables


<table>
<tr><th> Python (DFTBaby) </th><th> JSON (new) </th><th> Rust (new) </th></tr>
<tr><td>

| Attribute    | Type                        |
|--------------|-----------------------------|
| Dipole       | dict{tuple: ndarray[float]} |
| H            | dict{tuple: ndarray[float]} |
| S            | dict{tuple: ndarray[float]} |
| Z1           | int                         |
| Z2           | int                         |
| d            | ndarray[float]              |
| index2symbol | dict{int: str}              |

</td><td>

| Attribute       | Type                       |
|-----------------|----------------------------|
| dipole          | object{str: array[number]} |
| h               | object{str: array[number]} |
| s               | object{str: array[number]} |
| z1              | number                     |
| z2              | number                     |
| d               | list[float]                |
| index_to_symbol | object{number: string}     |

</td><td>

| Attribute       | Type             |
|-----------------|------------------|
| dipole          | dict             |
| h               | dict             |
| s               | dict             |
| z1              | u8               |
| z2              | u8               |
| d               | ndarray[float64] |
| index_to_symbol | dict             |

</td></tr> </table>


### Repulsive Pair Potential

| Attribute   | Type   | Type (depth=1)   |
|-------------|--------|------------------|
| vrep        | list   | float            |
| z1          | int    |                  |
| z2          | int    |                  |
| d           | list   | float            |