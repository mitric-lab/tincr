### Free Pseudo Atoms

#### Python types saved as json

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