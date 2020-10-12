## Umgang mit den DFTB Parametern

Dieser Abschnitt betrifft folgende Parameter: 
 - confined pseudo atoms
 - free pseudo atoms 
 - pairwise Slater-Koster tables
 - pairwise Repulsive-Potential tables
    
In DFTBaby werden die "homegrown" Parameter als Python Dateien (Modul) zur 
Verfügung gestellt und es gibt die Möglichkeit die Parameterdateien aus
DFTB+ oder Hotbit einzulesen. Da es in Rust wenig Sinn ergibt, Python Dateien 
zum Speichern von Daten zu benutzen, muss hier eine andere Lösung her. Ich hatte
zunächst auch  überlegt alle Parameter als Rust-Dateien zu schreiben und
zur Compile-Zeit vorliegen zu haben und in die Binary zu "backen". Allerdings ist
das Aufrufen der Daten dann etwas umständlich.  Es gibt auch zwei
ganz nützliche Artikel auf Github über das Speichern von Daten in Rust: 
[Global Data in Rust](https://github.com/paulkernfeld/global-data-in-rust)
und [Contiguous Data in Rust](https://github.com/paulkernfeld/contiguous-data-in-rust)

Aktuell halte ich es am für am sinnvollsten die Parameter zu Beginn der Laufzeit zu laden,
da die Datenmengen nicht sehr groß sind und dies effizient gehen sollte. Das Package
[serde](https://serde.rs) scheint dafür sehr gut geeignet zu sein. Dies erlaubt es effizient Daten 
aus verschiedenen Dateiformaten in Rust ´structs´ zu laden. 
Ich hatte zunächst mit JSON Dateien gearbeitet, diese bieten allerdings einige sehr lästige Nachteile. 
In JSON gibt es nur sehr wenige Datentypen (numbers, objects, arrays, strings), so dass es beispielsweise
nicht möglich ist ein Tuple als Key für ein Dictionary/Map zu benutzen. Dadurch lässt sich das nur durch 
die Benutzung von Strings umgehen und dies macht das Deserialisieren wiederum deutlich aufwendiger. 
Daher halte ich [Rusty Object Notation (RON)](https://github.com/ron-rs/ron) für das passendere Dateiformat. 
Dies ist analog zu JSON aufgebaut, enthält aber genau die benötigten Features. 
Als Einstieg für die Benutzung von serde in Kombination mit JSON Dateien fand ich dieses [Video](https://www.youtube.com/watch?v=hIi_UlyIPMg) hilfreich.
Infos zu Serdes default Deklaration, die ich oft benutze, finden sich hier: [Link](https://serde.rs/field-attrs.html)  

Die DFTB+ und Hotbit Parameter müsste man dann entweder manuell einlesen oder man konvertiert diese
zu JSON Dateien, um einheitliche Paramter-Dateiformate zu benutzen. 

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
