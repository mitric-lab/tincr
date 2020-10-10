### Free Pseudo Atoms

#### Python types saved as json
<table>
<tr><th> Python (DFTBaby) </th><th> JSON (new) </th><th> Rust (new) </th></tr>
<tr><td>

| Attribute            | Type              |
|----------------------|-------------------|
| Z                    | int               |
| Nelec               | int               |
| R0                   | float             |
| R                    | list[float]       |
| radial_density       | list[float]       |
| occupation           | list[tuple[int]]  |
| effective_potential  | list[float]       |
| orbital_names        | list[str]         |
| energies             | list[float]       |
| radial_wavefunctions | list[list[float]] |
| angular_momenta      | list[int]         |
| valence_orbitals     | list[int]         |
| energy_1s            | float             |
| orbital_1s           | list[float]       |

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


### Confined Pseudo Atoms

| Attribute            | Type              |
|----------------------|-------------------|
| z                    | int               |
| n_elec               | int               |
| r0                   | float             |
| r                    | list[float]       |
| radial_density       | list[float]       |
| occupation           | list[tuple[int]]  |
| effective_potential  | list[float]       |
| orbital_names        | list[str]         |
| energies             | list[float]       |
| radial_wavefunctions | list[list[float]] |
| angular_momenta      | list[int]         |
| valence_orbitals     | list[int]         |
| energy_1s            | float             |
| orbital_1s           | list[float]       |


### Slater-Koster tables

| Attribute    | Type   | Type (depth=1)   |
|--------------|--------|------------------|
| dipole       | dict   | str: list        |
| h            | dict   | str: list        |
| s            | dict   | str: list        |
| z1           | int    |                  |
| z2           | int    |                  |
| d            | list   | float            |
| index2symbol | dict   | int: str         |


### Repulsive Pair Potential

| Attribute   | Type   | Type (depth=1)   |
|-------------|--------|------------------|
| vrep        | list   | float            |
| z1          | int    |                  |
| z2          | int    |                  |
| d           | list   | float            |