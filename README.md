<div align="left">
  <img src="https://github.com/hochej/tincr/blob/master/tincr.svg" height="110"/>
</div>

Idee dieses Projekts ist das DFTBaby-Programmpaket von Alexander Humeniuk 
in Rust zu portieren. Die Vorteile dabei wären, dass die Effizienz deutlich
verbessert werden könnte und das Programm einfacher zu parallelisieren wäre.
Zudem sollen fragment-orbital basierende Rechnungen auf Basis von DFTB
implementiert werden (FMO Methode).

### Ideen zu den quantenchemischen Features
- Vorbild ist zunächst DFTBaby
- evtl. auch Gauss- (statt Slater)- Funktionen implementieren, um die DFTB+ Parameter nutzen zu können
- spin-unrestricted DFTB würde Beschreibung der Tripletts verbessern
- FMO-Methode auf Basis von DFTB-Monomer Rechnungen 
    - siehe FMO-DFTB: https://pubs.acs.org/doi/10.1021/ct500489d
    - siehe FMO-LC-DFTB: https://pubs.acs.org/doi/pdf/10.1021/acs.jctc.9b00108
- Spin-Bahn Kopplungen: https://pubs.acs.org/doi/10.1021/acs.jctc.6b00915
    - Implementierung https://github.com/jzpathfinder/pysoc
    

### Ein paar Ideen zum technischen Aufbau des Programms
- wenig OOP => Code sollte gut lesbar und leicht verständlich bleiben
- Atom-Klasse aus ASE könnte man vom Design übernehmen und selber implementieren.
 (siehe https://gitlab.com/ase/ase/-/blob/master/ase/atoms.py)


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
- bisher ist mir nicht klar wie man die Parameter am besten zur Verfügung
stellen sollte. Zurzeit denke ich, dass es am besten wäre alle Parameter 
zur Compile-Zeit als globale Daten vorliegen zu haben. 
Dieser Artikel hat dazu ganz interessante Infos: https://github.com/paulkernfeld/global-data-in-rust
Vielleicht bietet sich das [quote](https://crates.io/crates/quote) Paket an 
 - es würde eventuell zusätzlich Sinn machen, das Programm direkt kompatibel zu den [DFTB+ Parametern](https://dftb.org/parameters) zu machen,
 damit man automatisch auch auf diese Parameter zugreifen kann.

#### Effizienz
- zum Arbeiten mit Arrays und linearer Algebra scheint das [ndarray](https://docs.rs/ndarray/0.13.1/ndarray/) Paket gut geeignet zu sein
- vielleicht sollte auch das [smallvec](https://crates.io/crates/smallvec) Paket auch berücksichtigt werden 

#### Schnittstelle (API) zu Python
Ein Interface zu Python lässt sich recht einfach mit dem Modul [PyO3](https://github.com/PyO3/pyo3) implementieren. Ein konkretes Beispiel 
in Kombination mit numpy findet sich hier: https://github.com/PyO3/rust-numpy/tree/master/examples/simple-extension
Ein eigenes Beispiel von mir für solch ein Interface zu Python findet sich hier: https://github.com/hochej/tincr/README.md

-----------------------------------
### Einstieg in Rust
Anbei ein paar nützliche Seiten um Rust zu lernen (neben dem offiziellen Buch):


1. Amos Wegner's A half-hour to learn Rust: https://fasterthanli.me/articles/a-half-hour-to-learn-rust
2. Richard Anaya's Tour of Rust: https://tourofrust.com/00_en.html
3. Rustlings debugging Aufgaben: https://github.com/rust-lang/rustlings
4. Auf Excercism.io gibt es sehr viele Aufgaben zu Rust (leicht, mittel, schwer): https://exercism.io/tracks/rust. 
5. Tutorial um einen eigenen vim-artigen Text-Editor in Rust zu schreiben: https://www.philippflenker.com/hecto/

Sehr ausführliche Übersicht zu Rust (quasi ein Buch auf Github): https://github.com/Dhghomon/easy_rust/blob/master/README.md 

Infos zu Rust Modulen: http://www.sheshbabu.com/posts/rust-module-system/
Infos zu Lifetimes: https://github.com/pretzelhammer/rust-blog/blob/master/posts/common-rust-lifetime-misconceptions.md
Infos zu Sizedness: https://github.com/pretzelhammer/rust-blog/blob/master/posts/sizedness-in-rust.md
 
All diese Vorschläge sind aus dem offiziellen Rust-Forum entnommen: https://users.rust-lang.org/t/best-way-to-learn-rust-programming/47522/3