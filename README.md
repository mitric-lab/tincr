<div align="left">
  <img src="https://github.com/hochej/tincr/blob/master/tincr.png" height="120"/>
</div>
Idee dieses Projekts ist das DFTBaby-Programmpaket von Alexander Humeniuk 
in Rust zu portieren. Die Vorteile dabei wären, dass die Effizienz deutlich
verbessert werden könnte und das Programm einfacher zu parallelisieren wäre.
Zudem sollen fragment-orbital basierende Rechnungen auf Basis von DFTB
implementiert werden (FMO Methode).

### Ideen zu den quantenchemischen Features
- **Vorbild ist DFTBaby**
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
##### Input
- Command line input für xyz für Optimierung oder Single point Rechnung (simple Rechnungen)
- für alles andere Q-Chme-Input Format

##### Ouput
- der normale Output sollte mit einem allgemeinen Logger geschrieben werden. 
Das Paket [log](https://docs.rs/log/0.4.11/log/) könnte dafür passend sein.  
- das .traj Format von ASE wäre praktisch als Output-Format
für Optimierung/Dynamik (https://gitlab.com/ase/ase/-/blob/master/ase/io/trajectory.py)
- zusätzlicher Quantenchemie-Output könnte in ein .fchk-File geschrieben werden. Der Vorteil wäre hier,
dass das Format einfach zu schreiben ist und von vielen Programmen bereits gelesen werden kann
- Orbital-Informationen können in .molden-Files geschrieben werden wie das in DFTBaby bereits der Fall ist

#### Parameter
Die Beschreibung des Umgangs mit den Paramteren befindet sich auf der [extra Seite](https://github.com/hochej/tincr/blob/master/Parameter.md)

#### Einheiten und Konstanten
- die Verwaltung der Einheiten  könnte man vom Konzept von ASE übernehmen, siehe https://gitlab.com/ase/ase/-/blob/master/ase/units.py
- andere Konstanten könnte man als `const` oder `static` mit (statischer Lebenszeit) definieren 

#### Effizienz
- zum Arbeiten mit Arrays und linearer Algebra scheint das [ndarray](https://docs.rs/ndarray/0.13.1/ndarray/) Paket gut geeignet zu sein
- vielleicht sollte auch das [smallvec](https://crates.io/crates/smallvec) Paket auch berücksichtigt werden 

#### Speicherverwaltung 
- es müsste ein Check stattfinden, wie groß das Molekül ist und ob die Matrizen noch in den RAM passen,
ansonsten müssen die Arrays auf die Festplatte geschrieben werden
- anonsten könnte versucht werden viele Arrays nicht im gesamten zu Berechnen, sondern nur die Elemente 
sequentiell 

#### Schnittstelle (API) zu Python
Ein Interface zu Python lässt sich recht einfach mit dem Modul [PyO3](https://github.com/PyO3/pyo3) implementieren. Ein konkretes Beispiel 
in Kombination mit numpy findet sich hier: https://github.com/PyO3/rust-numpy/tree/master/examples/simple-extension <br>
Ein eigenes Beispiel von mir für solch ein Interface zu Python findet sich hier: https://github.com/hochej/tincr/blob/master/PyO3_Beispiel.md

#### Dokumentation
Ein paar Hinweise zur Dokumentation des Codes sind auf dieser [extra Seite](https://github.com/hochej/tincr/blob/master/Documentation.md) gegeben.
Kommentare die mit drei Slashes beginnen werden automatisch in der Dokumentation
als Text angezeigt. 


-----------------------------------
### Einstieg in Rust
Anbei ein paar nützliche Seiten um Rust zu lernen (neben dem offiziellen [Buch](https://doc.rust-lang.org/book/)):


1. [Amos Wegner's A half-hour to learn Rust](https://fasterthanli.me/articles/a-half-hour-to-learn-rust)
2. [Richard Anaya's Tour of Rust](https://tourofrust.com/00_en.html)
3. [Rustlings debugging Aufgaben](https://github.com/rust-lang/rustlings)
4. Auf [Excercism.io](https://exercism.io/tracks/rust.) gibt es sehr viele Aufgaben zu Rust (leicht, mittel, schwer)  
5. [Tutorial](https://www.philippflenker.com/hecto/) um einen eigenen vim-artigen Text-Editor in Rust zu schreiben

Sehr ausführliche [Übersicht](https://github.com/Dhghomon/easy_rust/blob/master/README.md ) zu Rust (quasi ein Buch auf Github)

[Infos zu Rust Modulen](http://www.sheshbabu.com/posts/rust-module-system/) <br>
[Infos zu Lifetimes](https://github.com/pretzelhammer/rust-blog/blob/master/posts/common-rust-lifetime-misconceptions.md) <br> 
[Infos zu Sizedness](https://github.com/pretzelhammer/rust-blog/blob/master/posts/sizedness-in-rust.md) <br>
 
All diese Vorschläge sind aus dem offiziellen [Rust-Forum](https://users.rust-lang.org/t/best-way-to-learn-rust-programming/47522/3
) entnommen.
#### Editor/IDE für Rust
Die IDE [CLion](https://www.jetbrains.com/clion/) von Jetbrains bietet eine sehr gute Unterstützung für Rust (durch ein Plugin). Zusätzlich kann man darüber
die git Versionierung sehr einfach durchführen. Für Angestellte an der Uni und Studierende ist das Paket [kostenlos](https://www.jetbrains.com/de-de/community/education/#students).

#### Rust naming conventions  

<p>In general, Rust tends to use <code>CamelCase</code> for &quot;type-level&quot; constructs
(types and traits) and <code>snake_case</code> for &quot;value-level&quot; constructs. More
precisely:</p>

<table><thead>
<tr>
<th>Item</th>
<th>Convention</th>
</tr>
</thead><tbody>
<tr>
<td>Crates</td>
<td><code>snake_case</code> (but prefer single word)</td>
</tr>
<tr>
<td>Modules</td>
<td><code>snake_case</code></td>
</tr>
<tr>
<td>Types</td>
<td><code>CamelCase</code></td>
</tr>
<tr>
<td>Traits</td>
<td><code>CamelCase</code></td>
</tr>
<tr>
<td>Enum variants</td>
<td><code>CamelCase</code></td>
</tr>
<tr>
<td>Functions</td>
<td><code>snake_case</code></td>
</tr>
<tr>
<td>Methods</td>
<td><code>snake_case</code></td>
</tr>
<tr>
<td>General constructors</td>
<td><code>new</code> or <code>with_more_details</code></td>
</tr>
<tr>
<td>Conversion constructors</td>
<td><code>from_some_other_type</code></td>
</tr>
<tr>
<td>Local variables</td>
<td><code>snake_case</code></td>
</tr>
<tr>
<td>Static variables</td>
<td><code>SCREAMING_SNAKE_CASE</code></td>
</tr>
<tr>
<td>Constant variables</td>
<td><code>SCREAMING_SNAKE_CASE</code></td>
</tr>
<tr>
<td>Type parameters</td>
<td>concise <code>CamelCase</code>, usually single uppercase letter: <code>T</code></td>
</tr>
<tr>
<td>Lifetimes</td>
<td>short, lowercase: <code>&#39;a</code></td>
</tr>
</tbody></table>