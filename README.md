<div align="left">
  <img src="https://github.com/hochej/tincr/blob/master/tincr.svg" height="110"/>
</div>

### tincr
Idee dieses Projekts ist das DFTBaby-Programmpaket von A. Humeniuk 
in Rust zu portieren. Die Vorteile dabei wären, dass die Effizienz deutlich
verbessert werden könnte und das Programm einfacher zu parallelisieren wäre.
Zudem sollen fragment-orbital basierende Rechnungen auf Basis von DFTB
implementiert werden (FMO Methode).

Folgende Anforderungen sollte das Programm haben:
- möglichst wenig OOP, nur absolut notwendige Klassen/structs implementieren
(eventuell für Moleküle und Atome)
- bisher ist mir nicht klar wie man die Parameter am besten zur Verfügung
stellen sollte
- es sollte ein einfaches und einheitliches Format für den Input geben,
dabei könnte das Q-Chem Format vielleicht als Vorbild dienen
- zu jedem Abschnitt sollten unit-tests implementiert werden, um die Ergebnisse
der Funktionen zu testen 