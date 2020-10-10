### Free Pseudo Atoms

#### Python types saved as json

| Attribute            | Type              |       | Attribute            | Type              |
|----------------------|-------------------|       |----------------------|-------------------|
| z                    | int               |       | z                    | number               |
| n_elec               | int               |       | n_elec               | number               |
| r0                   | float             |       | r0                   | number             |
| r                    | list[float]       |       | r                    | array[number]       |
| radial_density       | list[float]       |       | radial_density       | array[number]       |
| occupation           | list[tuple[int]]  |       | occupation           | array[tuple[number]]  |
| effective_potential  | list[float]       |       | effective_potential  | array[number]       |
| orbital_names        | list[str]         |       | orbital_names        | array[string]         |
| energies             | list[float]       |       | energies             | array[number]       |
| radial_wavefunctions | list[list[float]] |       | radial_wavefunctions | array[array[number]] |
| angular_momenta      | list[int]         |       | angular_momenta      | array[number]         |
| valence_orbitals     | list[int]         |       | valence_orbitals     | array[number]         |
| energy_1s            | float             |       | energy_1s            | number             |
| orbital_1s           | list[float]       |       | orbital_1s           | array[number]       |


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