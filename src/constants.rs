use phf::{phf_map, phf_set};

pub const BOHR_TO_ANGS: f64 = 0.529177249;
pub const HARTREE_TO_EV: f64 = 27.211396132;
pub const HARTREE_TO_NM: f64 = 45.563352527; // lambda (in nm) = HARTREE_TO_NM / (energy in Hartree)
pub const HARTREE_TO_WAVENUMBERS: f64 = 219474.63; // # E(in cm^-1) = E(in Hartree) * HARTREE_TO_WAVENUMBERS
pub const HARTREE_TO_KCALMOL: f64 = 627.509469;
pub const AUTIME2FS: f64 = 0.02418884326505;
pub const K_BOLTZMANN: f64 = 3.1668114 * 1.0e-6; // in hartree/Kelvin
pub const AUMASS2AMU: f64 = 1.0 / 1822.888486192; // convert masses from atomic units to amu
pub const EBOHR_TO_DEBYE: f64 = 1.0 / 0.393430307; // 1 Debye = 0.393430307 e*a0
pub const SPEED_OF_LIGHT: f64 = 137.035999139; // speed of light in atomic units, inverse of fine structure constant
                                               // c = 1/alpha

// THESE ARE NO REAL CONSTANTS, THEY ARE DEFAULT VALUES
pub const LONG_RANGE_RADIUS: f64 = 3.0300;
pub const PROXIMITY_CUTOFF: f64 = 2.00;
pub const DEFAULT_CHARGE: i8 = 0;
pub const DEFAULT_MULTIPLICITY: u8 = 1;
pub const DEFAULT_MAX_ITER: usize = 250;
pub const DEFAULT_SCF_CONV: f64 = 1.0e-7;
pub const DEFAULT_TEMPERATURE: f64 = 0.0;

pub const ATOM_NAMES: [&str; 86] = [
    "h", "he", "li", "be", "b", "c", "n", "o", "f", "ne", "na", "mg", "al", "si", "p", "s", "cl",
    "ar", "k", "ca", "sc", "ti", "v", "cr", "mn", "fe", "co", "ni", "cu", "zn", "ga", "ge", "as",
    "se", "br", "kr", "rb", "sr", "y", "zr", "nb", "mo", "tc", "ru", "rh", "pd", "ag", "cd", "in",
    "sn", "sb", "te", "i", "xe", "cs", "ba", "la", "ce", "pr", "nd", "pm", "sm", "eu", "gd", "tb",
    "dy", "ho", "er", "th", "yt", "lu", "hf", "ta", "w", "re", "os", "ir", "pt", "au", "hg", "tl",
    "pb", "bi", "po", "at", "rn",
];

// I believe these masses are averaged over isotopes weighted with their abundances
pub static ATOMIC_MASSES: phf::Map<&'static str, f64> = phf_map! {
    "h"  => 1.837362128065067E+03, "he" => 7.296296732461748E+03,
    "li" => 1.265266834424631E+04, "be" => 1.642820197435333E+04,
    "b"  => 1.970724642985837E+04, "c"  => 2.189416563639810E+04,
    "n"  => 2.553265087125124E+04, "o"  => 2.916512057440347E+04,
    "f"  => 3.463378575704848E+04, "ne" => 3.678534092874044E+04,
    "na" => 4.190778360619254E+04, "mg" => 4.430530242139558E+04,
    "al" => 4.918433357200404E+04, "si" => 5.119673199572539E+04,
    "p"  => 5.646171127497759E+04, "s"  => 5.845091636050397E+04,
    "cl" => 6.462686224010440E+04, "ar" => 7.282074557210083E+04,
    "k"  => 7.127183730353635E+04, "ca" => 7.305772106334878E+04,
    "sc" => 8.194961023615085E+04, "ti" => 8.725619876588941E+04,
    "v"  => 9.286066913390342E+04, "cr" => 9.478308723444256E+04,
    "mn" => 1.001459246313614E+05, "fe" => 1.017992023749367E+05,
    "co" => 1.074286371995095E+05, "ni" => 1.069915176770187E+05,
    "cu" => 1.158372658987864E+05, "zn" => 1.191804432137767E+05,
    "ga" => 1.270972475098524E+05, "ge" => 1.324146129557776E+05,
    "as" => 1.365737151160185E+05, "se" => 1.439352676072164E+05,
    "br" => 1.456560742513554E+05, "kr" => 1.527544016584286E+05,
    "rb" => 1.557982606990888E+05, "sr" => 1.597214811011183E+05,
    "y"  => 1.620654421428197E+05, "zr" => 1.662911708738692E+05,
    "nb" => 1.693579618505286E+05, "mo" => 1.749243703088714E+05,
    "tc" => 1.786430626330700E+05, "ru" => 1.842393300033101E+05,
    "rh" => 1.875852416508917E+05, "pd" => 1.939917829123603E+05,
    "ag" => 1.966316898848625E+05, "cd" => 2.049127072821024E+05,
    "in" => 2.093003996469779E+05, "sn" => 2.163950812772627E+05,
    "sb" => 2.219548908796184E+05, "te" => 2.326005591018340E+05,
    "i"  => 2.313326855370057E+05, "xe" => 2.393324859416700E+05,
    "cs" => 2.422718057964099E+05, "ba" => 2.503317945123633E+05,
    "la" => 2.532091691559799E+05, "ce" => 2.554158302438290E+05,
    "pr" => 2.568589198411092E+05, "nd" => 2.629370677583600E+05,
    "pm" => 2.643188171611750E+05, "sm" => 2.740894989541674E+05,
    "eu" => 2.770134119384883E+05, "gd" => 2.866491999903088E+05,
    "tb" => 2.897031760615568E+05, "dy" => 2.962195463487770E+05,
    "ho" => 3.006495661821661E+05, "er" => 3.048944899280067E+05,
    "tm" => 3.079482107948796E+05, "yb" => 3.154581281724826E+05,
    "lu" => 3.189449490929371E+05, "hf" => 3.253673494834354E+05,
    "ta" => 3.298477904098085E+05, "w"  => 3.351198023924856E+05,
    "re" => 3.394345792215925E+05, "os" => 3.467680592315195E+05,
    "ir" => 3.503901384708247E+05, "pt" => 3.501476943143941E+05,
    "au" => 3.590480726784480E+05, "hg" => 3.656531829955869E+05,
    "tl" => 3.725679455413626E+05, "pb" => 3.777024752813480E+05,
    "bi" => 3.809479476012968E+05, "po" => 3.828065627851500E+05,
    "at" => 3.828065627851500E+05, "rn" => 4.010354467273000E+05
};


// //  occupation numbers of valence orbitals
// //  which are used to assign the correct occupation to orbitals loaded from hotbit .elm files
// pub static OCCUPATION_NUMBERS: phf::Map<&'static str, f64> = phf_map! {
// "h" => {"1s": 1},
// "he" => {"1s": 2},
// "li" => {"2s": 1},
// "be" => {"2s": 2},
// "b" => {"2s": 2, "2p": 1},
// "c" => {"2s": 2, "2p": 2},
// "n" => {"2s": 2, "2p": 3},
// "o" => {"2s": 2, "2p": 4},
// "f" => {"2s": 2, "2p": 5},
// "ne" => {"2s": 2, "2p": 6},
// "na" => {"3s": 1}, "mg": {"3s": 2},
// "al" => {"3s": 2, "3p": 1},
// "si" => {"3s": 2, "3p": 2},
// "p": {"3s": 2, "3p": 3},
// "s": {"3s": 2, "3p": 4},
// "cl": {"3s": 2, "3p": 5},
// "ar": {"3s": 2, "3p": 6},
// "ti": {"3d": 2, "4s": 2},
// "zn": {"3d": 10, "4s": 2},
// "br": {"4s": 2, "4p": 5},
// "ru": {"4d": 7, "5s": 1},
// "i": {"5s": 2, "5p": 5},
// "au": {"4f": 14, "5d": 10, "6s": 1}
// };