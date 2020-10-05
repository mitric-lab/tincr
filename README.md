<div align="left">
  <img src="https://github.com/hochej/tincr/blob/master/tincr.svg" height="110"/>
</div>

Idee dieses Projekts ist das DFTBaby-Programmpaket von A. Humeniuk 
in Rust zu portieren. Die Vorteile dabei wären, dass die Effizienz deutlich
verbessert werden könnte und das Programm einfacher zu parallelisieren wäre.
Zudem sollen fragment-orbital basierende Rechnungen auf Basis von DFTB
implementiert werden (FMO Methode).

Folgende Anforderungen sollte das Programm haben:
- möglichst wenig OOP, nur absolut notwendige Klassen/structs implementieren
(eventuell für Moleküle und Atome). Vielleicht würde es Sinn machen die 
Atom-Klasse aus ASE vom Design her zu übernehmen und selber zu implementieren. 
- bisher ist mir nicht klar wie man die Parameter am besten zur Verfügung
stellen sollte. Zurzeit denke ich, dass es am besten wäre alle Parameter 
zur Compile-Zeit als globale Daten vorliegen zu haben. 
Dieser Artikel hat dazu ganz interessante Infos: https://github.com/paulkernfeld/global-data-in-rust
- es sollte ein einfaches und einheitliches Format für den Input geben,
dabei könnte das Q-Chem Format vielleicht als Vorbild dienen
- zu jedem Abschnitt sollten unit-tests implementiert werden, um die Ergebnisse
der Funktionen zu testen 


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