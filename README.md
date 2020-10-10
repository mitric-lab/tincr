<div align="left">
  <img src="https://github.com/hochej/tincr/blob/master/tincr.png" height="120"/>
</div>
Idee dieses Projekts ist das DFTBaby-Programmpaket von Alexander Humeniuk 
in Rust zu portieren. Die Vorteile dabei wären, dass die Effizienz deutlich
verbessert werden könnte und das Programm einfacher zu parallelisieren wäre.
Zudem sollen fragment-orbital basierende Rechnungen auf Basis von DFTB
implementiert werden (FMO Methode).

### Ideen zu den quantenchemischen Features
- Vorbild ist zunächst DFTBaby
- sowohl Gauss- als auch Slater-Funktionen implementieren, um die DFTB+ Parameter nutzen zu können
- spin-unrestricted DFTB würde Beschreibung der Tripletts verbessern
- könnte man eine Parametrisierung anhand des ANI-Datensatzes durchführen?
- https://aip.scitation.org/doi/pdf/10.1063/5.0020545
- FMO-Methode auf Basis von DFTB-Monomer Rechnungen 
    - siehe FMO-DFTB: https://pubs.acs.org/doi/10.1021/ct500489d
    - siehe FMO-LC-DFTB: https://pubs.acs.org/doi/pdf/10.1021/acs.jctc.9b00108
- Spin-Bahn Kopplungen: https://pubs.acs.org/doi/10.1021/acs.jctc.6b00915
    - Implementierung https://github.com/jzpathfinder/pysoc
    

### Ein paar Ideen zum technischen Aufbau des Programms
- wenig OOP => Code sollte gut lesbar und leicht verständlich bleiben
- [Atoms-Klasse aus ASE](https://gitlab.com/ase/ase/-/blob/master/ase/atoms.py) oder [mole-Klasse aus PySCF](https://github.com/pyscf/pyscf/blob/532842f439cd1cffc7fa61749fffb9879bbc92c9/pyscf/gto/mole.py#L1899)
 könnte man vom Design übernehmen und selber implementieren.
 


#### Input/Output-Format
- es sollte ein einfaches und einheitliches Format für den Input geben,
dabei könnte das Q-Chem Format vielleicht als Vorbild dienen
- das .traj Format von ASE wäre eventuell praktisch als Output-Format
für Optimierung/Dynamik (https://gitlab.com/ase/ase/-/blob/master/ase/io/trajectory.py)
- der normale Output sollte vielleicht mit einem allgemeinen Logger geschrieben werden. 
Das Paket [log](https://docs.rs/log/0.4.11/log/) könnte dafür passend sein.  
- zusätzlicher Quantenchemie-Output könnte in ein .fchk-File geschrieben werden. Der Vorteil wäre hier,
dass das Format einfach zu schreiben ist und von vielen Programmen bereits gelesen werden kann
- Orbital-Informationen können auch gut in .molden-Files geschrieben werden wie das in DFTBaby bereits der Fall ist

#### Parameter
##### Einheiten und Konstanten
- die Verwaltung der Einheiten  könnte man vom Konzept von ASE übernehmen, siehe https://gitlab.com/ase/ase/-/blob/master/ase/units.py
- andere Konstanten könnte man als `const` oder `static` mit (statischer Lebenszeit) definieren 
##### DFTB Parameter
Dieser Absatz betrifft folgende Parameter: 
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
aus verschiedenen Dateiformaten in Rust ´structs´ zu laden. Ich halte ein
Format wie JSON dafür am geeignetsten, da es sehr leicht zu schreiben und auch 
lesbar im nachhinein ist. Als Einstieg für die Benutzung von serde in Kombination
mit JSON Dateien fand ich dieses [Video](https://www.youtube.com/watch?v=hIi_UlyIPMg) hilfreich.  

Die DFTB+ und Hotbit Parameter müsste man dann entweder manuell einlesen oder man konvertiert diese
zu JSON Dateien, um einheitliche Paramter-Dateiformate zu benutzen. 

#####  
Dieser Artikel hat dazu ganz interessante Infos: 
Vielleicht bietet sich das [quote](https://crates.io/crates/quote) Paket an 
 - es würde eventuell zusätzlich Sinn machen, das Programm direkt kompatibel zu den [DFTB+ Parametern](https://dftb.org/parameters) zu machen,
 damit man automatisch auch auf diese Parameter zugreifen kann.

#### Effizienz
- zum Arbeiten mit Arrays und linearer Algebra scheint das [ndarray](https://docs.rs/ndarray/0.13.1/ndarray/) Paket gut geeignet zu sein
- vielleicht sollte auch das [smallvec](https://crates.io/crates/smallvec) Paket auch berücksichtigt werden 

#### Schnittstelle (API) zu Python
Ein Interface zu Python lässt sich recht einfach mit dem Modul [PyO3](https://github.com/PyO3/pyo3) implementieren. Ein konkretes Beispiel 
in Kombination mit numpy findet sich hier: https://github.com/PyO3/rust-numpy/tree/master/examples/simple-extension <br>
Ein eigenes Beispiel von mir für solch ein Interface zu Python findet sich hier: https://github.com/hochej/tincr/blob/master/PyO3_Beispiel.md

-----------------------------------
### Einstieg in Rust
Anbei ein paar nützliche Seiten um Rust zu lernen (neben dem offiziellen [Buch](https://doc.rust-lang.org/book/)):


1. [Amos Wegner's A half-hour to learn Rust](https://fasterthanli.me/articles/a-half-hour-to-learn-rust)
2. [Richard Anaya's Tour of Rust](https://tourofrust.com/00_en.html)
3. [Rustlings debugging Aufgaben](https://github.com/rust-lang/rustlings)
4. Auf [Excercism.io](https://exercism.io/tracks/rust.) gibt es sehr viele Aufgaben zu Rust (leicht, mittel, schwer)  
5. [Tutorial](https://www.philippflenker.com/hecto/) um einen eigenen vim-artigen Text-Editor in Rust zu schreiben

Sehr ausführliche [Übersicht](https://github.com/Dhghomon/easy_rust/blob/master/README.md ) zu Rust (quasi ein Buch auf Github)

[Infos zu Rust Modulen](http://www.sheshbabu.com/posts/rust-module-system/) 
[Infos zu Lifetimes](https://github.com/pretzelhammer/rust-blog/blob/master/posts/common-rust-lifetime-misconceptions.md) 
[Infos zu Sizedness](https://github.com/pretzelhammer/rust-blog/blob/master/posts/sizedness-in-rust.md) 
 
All diese Vorschläge sind aus dem offiziellen [Rust-Forum](https://users.rust-lang.org/t/best-way-to-learn-rust-programming/47522/3
) entnommen.
#### Editor/IDE für Rust
Die IDE [CLion](https://www.jetbrains.com/clion/) von Jetbrains bietet eine sehr gute Unterstützung für Rust (durch ein Plugin). Zusätzlich kann man darüber
die git Versionierung sehr einfach durchführen. Für Angestellte an der Uni und Studierende ist das Paket [kostenlos](https://www.jetbrains.com/de-de/community/education/#students). 