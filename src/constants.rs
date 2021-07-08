use phf::{phf_map, phf_set};

pub const BOHR_TO_ANGS: f64 = 0.529177210903;
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

pub const ATOM_NAMES: [&str; 87] = [
    "dummy", "h", "he", "li", "be", "b", "c", "n", "o", "f", "ne", "na", "mg", "al", "si", "p",
    "s", "cl", "ar", "k", "ca", "sc", "ti", "v", "cr", "mn", "fe", "co", "ni", "cu", "zn", "ga",
    "ge", "as", "se", "br", "kr", "rb", "sr", "y", "zr", "nb", "mo", "tc", "ru", "rh", "pd", "ag",
    "cd", "in", "sn", "sb", "te", "i", "xe", "cs", "ba", "la", "ce", "pr", "nd", "pm", "sm", "eu",
    "gd", "tb", "dy", "ho", "er", "th", "yt", "lu", "hf", "ta", "w", "re", "os", "ir", "pt", "au",
    "hg", "tl", "pb", "bi", "po", "at", "rn",
];

pub const TAUSYMBOLS_AB: [&str;10] = ["dd_sigma", "dd_pi", "dd_delta",   "pd_sigma", "pd_pi", "pp_sigma", "pp_pi", "sd_sigma", "sp_sigma", "ss_sigma"];
pub const TAUSYMBOLS_BA: [&str;10] = ["dd_sigma", "dd_pi", "dd_delta",   "dp_sigma", "dp_pi", "pp_sigma", "pp_pi", "ds_sigma", "ps_sigma", "ss_sigma"];

pub static SYMBOL_2_TAU:phf::Map<&'static str,(u8,i32,u8,i32)> = phf_map!{
    "ss_sigma" => (0,0,0,0),
    "sp_sigma" => (0,0,1,0),
    "ps_sigma" => (1,0,0,0),
    "pp_pi"    => (1,-1,1,-1),
    "pp_sigma" => (1,0,1,0),
    "ds_sigma" => (2,0,0,0),
    "sd_sigma" => (0,0,2,0),
    "dp_pi"    => (2,-1,1,-1),
    "pd_pi"    => (1,-1,2,-1),
    "dp_sigma" => (2,0,1,0),
    "pd_sigma" => (1,0,2,0),
    "dd_delta" => (2,-2,2,-2),
    "dd_pi"    => (2,-1,2,-1),
    "dd_sigma" => (2,0,2,0),
};

pub static ELEMENT_TO_Z: phf::Map<&'static str, u8> = phf_map! {
    "h"  => 1, "he" => 2,
    "li" => 3, "be" => 4,
    "b"  => 5, "c"  => 6,
    "n"  => 7, "o"  => 8,
    "f"  => 9, "ne" => 10,
};
// I believe these masses are averaged over isotopes weighted with their abundances
pub static ATOMIC_MASSES_OLD: phf::Map<&'static str, f64> = phf_map! {
    "H"  => 1.837362128065067E+03, "he" => 7.296296732461748E+03,
    "li" => 1.265266834424631E+04, "be" => 1.642820197435333E+04,
    "b"  => 1.970724642985837E+04, "C"  => 2.189416563639810E+04,
    "N"  => 2.553265087125124E+04, "O"  => 2.916512057440347E+04,
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

pub static ATOMIC_MASSES: phf::Map<u8, f64> = phf_map! {
    1u8  => 1.837362128065067E+03, 2u8 => 7.296296732461748E+03,
    3u8 => 1.265266834424631E+04, 4u8 => 1.642820197435333E+04,
    5u8  => 1.970724642985837E+04, 6u8  => 2.189416563639810E+04,
    7u8  => 2.553265087125124E+04, 8u8  => 2.916512057440347E+04,
    9u8 => 3.463378575704848E+04, 10u8 => 3.678534092874044E+04,
    11u8 => 4.190778360619254E+04, 12u8 => 4.430530242139558E+04,
    13u8 => 4.918433357200404E+04, 14u8 => 5.119673199572539E+04,
    15u8  => 5.646171127497759E+04, 16u8  => 5.845091636050397E+04,
    17u8 => 6.462686224010440E+04, 18u8 => 7.282074557210083E+04,
    19u8  => 7.127183730353635E+04, 20u8 => 7.305772106334878E+04,
    21u8 => 8.194961023615085E+04, 22u8 => 8.725619876588941E+04,
    23u8  => 9.286066913390342E+04, 24u8 => 9.478308723444256E+04,
    25u8 => 1.001459246313614E+05, 26u8 => 1.017992023749367E+05,
    27u8 => 1.074286371995095E+05, 28u8 => 1.069915176770187E+05,
    29u8 => 1.158372658987864E+05, 30u8 => 1.191804432137767E+05,
    31u8 => 1.270972475098524E+05, 32u8 => 1.324146129557776E+05,
    33u8 => 1.365737151160185E+05, 34u8 => 1.439352676072164E+05,
    35u8 => 1.456560742513554E+05, 36u8 => 1.527544016584286E+05,
    37u8 => 1.557982606990888E+05, 38u8 => 1.597214811011183E+05,
    39u8 => 1.620654421428197E+05, 40u8 => 1.662911708738692E+05,
    41u8 => 1.693579618505286E+05, 42u8 => 1.749243703088714E+05,
    43u8 => 1.786430626330700E+05, 44u8 => 1.842393300033101E+05,
    45u8 => 1.875852416508917E+05, 46u8 => 1.939917829123603E+05,
    47u8 => 1.966316898848625E+05, 48u8 => 2.049127072821024E+05,
    49u8 => 2.093003996469779E+05, 50u8 => 2.163950812772627E+05,
    51u8 => 2.219548908796184E+05, 52u8 => 2.326005591018340E+05,
    53u8  => 2.313326855370057E+05, 54u8 => 2.393324859416700E+05,
    55u8 => 2.422718057964099E+05, 56u8 => 2.503317945123633E+05,
    57u8 => 2.532091691559799E+05, 58u8 => 2.554158302438290E+05,
    59u8 => 2.568589198411092E+05, 60u8 => 2.629370677583600E+05,
    61u8 => 2.643188171611750E+05, 62u8 => 2.740894989541674E+05,
    63u8 => 2.770134119384883E+05, 64u8 => 2.866491999903088E+05,
    65u8 => 2.897031760615568E+05, 66u8 => 2.962195463487770E+05,
    67u8 => 3.006495661821661E+05, 68u8 => 3.048944899280067E+05,
    69u8 => 3.079482107948796E+05, 70u8 => 3.154581281724826E+05,
    71u8 => 3.189449490929371E+05, 72u8 => 3.253673494834354E+05,
    73u8 => 3.298477904098085E+05, 74u8  => 3.351198023924856E+05,
    75u8 => 3.394345792215925E+05, 76u8 => 3.467680592315195E+05,
    77u8 => 3.503901384708247E+05, 78u8 => 3.501476943143941E+05,
    79u8 => 3.590480726784480E+05, 80u8 => 3.656531829955869E+05,
    81u8 => 3.725679455413626E+05, 82u8 => 3.777024752813480E+05,
    83u8 => 3.809479476012968E+05, 84u8 => 3.828065627851500E+05,
    85u8 => 3.828065627851500E+05, 86u8 => 4.010354467273000E+05
};

// Spin coupling constants taken from dftb+ manual
pub static SPIN_COUPLING: phf::Map<u8, f64> = phf_map! {
    1u8  => -0.072,
    6u8  => -0.023,
    7u8  => -0.026,
    8u8  => -0.028,
};

// taken from
// Beatriz Cordero, Ver ́onica G ́omez, Ana E. Platero-Prats, Marc Rev ́es, Jorge Echeverr ́ıa, Eduard Cremades,Flavia Barrag ́an and Santiago Alvarez
// Covalent radii revisited, Dalton Trans., 2008, 2832–2838
pub static COVALENCE_RADII: phf::Map<u8, f64> = phf_map! {
    1u8 =>0.31,
    2u8 =>0.28,
    3u8 =>1.28,
    4u8 =>0.96,
    5u8 =>0.84,
    6u8 =>0.76,
    7u8 =>0.71,
    8u8 =>0.66,
    9u8 =>0.57,
    10u8 =>0.58,
    11u8 =>1.66,
    12u8 =>1.41,
    13u8 =>1.21,
    14u8 =>1.11,
    15u8 =>1.07,
    16u8 =>1.05,
    17u8 =>1.02,
    18u8 =>1.06,
    19u8 =>2.03,
    20u8 =>1.76,
    21u8 =>1.7,
    22u8 =>1.6,
    23u8 =>1.53,
    24u8 =>1.39,
    25u8 =>1.61,
    26u8 =>1.52,
    27u8 =>1.5,
    28u8 =>1.24,
    29u8 =>1.32,
    30u8 =>1.22,
    31u8 =>1.22,
    32u8 =>1.2,
    33u8 =>1.19,
    34u8 =>1.2,
    35u8 =>1.2,
    36u8 =>1.16,
    37u8 =>2.2,
    38u8 =>1.95,
    39u8 =>1.9,
    40u8 =>1.75,
    41u8 =>1.64,
    42u8 =>1.54,
    43u8 =>1.47,
    44u8 =>1.46,
    45u8 =>1.42,
    46u8 =>1.39,
    47u8 =>1.45,
    48u8 =>1.44,
    49u8 =>1.42,
    50u8 =>1.39,
    51u8 =>1.39,
    52u8 =>1.38,
    53u8 =>1.39,
    54u8 =>1.4,
    55u8 =>2.44,
    56u8 =>2.15,
    57u8 =>2.07,
    58u8 =>2.04,
    59u8 =>2.03,
    60u8 =>2.01,
    61u8 =>1.99,
    62u8 =>1.98,
    63u8 =>1.98,
    64u8 =>1.96,
    65u8 =>1.94,
    66u8 =>1.92,
    67u8 =>1.92,
    68u8 =>1.89,
    69u8 =>1.9,
    70u8 =>1.87,
    71u8 =>1.87,
    72u8 =>1.75,
    73u8 =>1.7,
    74u8 =>1.62,
    75u8 =>1.51,
    76u8 =>1.44,
    77u8 =>1.41,
    78u8 =>1.36,
    79u8 =>1.36,
    80u8 =>1.32,
    81u8 =>1.45,
    82u8 =>1.46,
    83u8 =>1.48,
    84u8 =>1.4,
    85u8 =>1.5,
    86u8 =>1.5,
    87u8 =>2.6,
    88u8 =>2.21,
    89u8 =>2.15,
    90u8 =>2.06,
    91u8 =>2.0,
    92u8 =>1.96,
    93u8 =>1.9,
    94u8 =>1.87,
    95u8 =>1.8,
    96u8 =>1.69
};

// A. Bondi
// The Journal of Physical Chemistry 1964 68 (3), 441-451
// For H:
// Rowland,  R.S.  and  Taylor,  R.,  Intermolecular  Non-bonded Contact Distances in Organic Crystal Structures:Comparison  with  Distances  Expected  from  van  derWaals  Radii,  J.  Phys.  Chem.,  1996,  vol.  100,  no.  18,pp. 7384–7391.
//
// different methods: Batsanov, S.S. Van der Waals Radii of Elements. Inorganic Materials 37, 871–885 (2001)
// Manjeera Mantina, Adam C. Chamberlin, Rosendo Valero, Christopher J. Cramer, and Donald G. Truhlar
// The Journal of Physical Chemistry A 2009 113 (19), 5806-5812

pub static VDW_RADII: phf::Map<u8, f64> = phf_map! {
    1u8 =>1.10,
    // 1u8 => 1.4430662541324808,
    2u8 =>1.40,
    6u8 =>1.70,
    7u8 =>1.55,
    // 8u8 =>1.52,
    8u8 => 1.8002608714920059,
    9u8 =>1.47,
    10u8 =>1.54,
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
