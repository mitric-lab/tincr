<div align="left">
  <img src="https://github.com/hochej/tincr/blob/master/tincr.svg" height="110"/>
</div>

Idee dieses Projekts ist das DFTBaby-Programmpaket von A. Humeniuk 
in Rust zu portieren. Die Vorteile dabei wären, dass die Effizienz deutlich
verbessert werden könnte und das Programm einfacher zu parallelisieren wäre.
Zudem sollen fragment-orbital basierende Rechnungen auf Basis von DFTB
implementiert werden (FMO Methode).

### Ein paar Ideen zur technischen Aufbau des Programms
- wenig OOP => Code sollte gut lesbar und leicht verständlich bleiben
- Atom-Klasse aus ASE könnte man vom Design übernehmen und selber implementieren.
 (siehe https://gitlab.com/ase/ase/-/blob/master/ase/atoms.py)


#### Input/Output-Format
- es sollte ein einfaches und einheitliches Format für den Input geben,
dabei könnte das Q-Chem Format vielleicht als Vorbild dienen
- das .traj Format von ASE wäre eventuell praktisch als Output-Format
für Optimierung/Dynamik (https://gitlab.com/ase/ase/-/blob/master/ase/io/trajectory.py)
- der normale Output sollte vielleicht mit einem allgemeinen Logger geschrieben werden. 
Das Paket https://docs.rs/log/0.4.11/log/ könnte dafür passend sein.  

#### Parameter
##### Einheiten und Konstanten
- die Verwaltung der Einheiten (https://gitlab.com/ase/ase/-/blob/master/ase/units.py) könnte man vom Konzept
von ASE übernehmen
- andere Konstanten könnte man als `const` oder `static` mit (statischer Lebenszeit) definieren 
##### DFTB Parameter
- bisher ist mir nicht klar wie man die Parameter am besten zur Verfügung
stellen sollte. Zurzeit denke ich, dass es am besten wäre alle Parameter 
zur Compile-Zeit als globale Daten vorliegen zu haben. 
Dieser Artikel hat dazu ganz interessante Infos: https://github.com/paulkernfeld/global-data-in-rust
Vielleicht bietet sich auch sowas an: https://crates.io/crates/quote
 - es würde eventuell zusätzlich Sinn machen, das Programm direkt kompatibel zu den DFTB+ Paramtern zu machen,
 damit man automatisch auch auf diese Parameter zugreifen kann.

#### Effizienz
- vielleicht sollte dies Paket berücksichtigt werden: https://crates.io/crates/smallvec 
-----------------------------------
### Einstieg in Rust
Anbei ein paar nützliche Seiten um Rust zu lernen (neben dem offiziellen Buch):


1. Amos Wegner's A half-hour to learn Rust: https://fasterthanli.me/articles/a-half-hour-to-learn-rust
2. Richard Anaya's Tour of Rust: https://tourofrust.com/00_en.html
3. Rustlings debugging Aufgaben: https://github.com/rust-lang/rustlings
4. Auf Excercism.io gibt es sehr viele Aufgaben zu Rust (leicht, mittel, schwer): https://exercism.io/tracks/rust. 
5. Tutorial um einen eigenen vim-artigen Text-Editor in Rust zu schreiben: https://www.philippflenker.com/hecto/

Sehr ausführliche Übersicht zu Rust: https://github.com/Dhghomon/easy_rust/blob/master/README.md 

Infos zu Rust Modulen: http://www.sheshbabu.com/posts/rust-module-system/
Infos zu Lifetimes: https://github.com/pretzelhammer/rust-blog/blob/master/posts/common-rust-lifetime-misconceptions.md
Infos zu Sizedness: https://github.com/pretzelhammer/rust-blog/blob/master/posts/sizedness-in-rust.md
 
All diese Vorschläge sind aus dem offiziellen Rust-Forum entnommen: https://users.rust-lang.org/t/best-way-to-learn-rust-programming/47522/3