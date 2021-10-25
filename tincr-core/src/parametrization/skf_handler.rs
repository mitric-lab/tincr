use crate::Element;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Clone)]
pub struct SkfHandler {
    pub el_a: Element,
    pub el_b: Element,
    pub data_string: String,
}

impl SkfHandler {
    pub fn new(el_a: Element, el_b: Element, path_prefix: &str) -> SkfHandler {
        let filename: String = format!("{}/{}-{}.skf", path_prefix, el_a.symbol(), el_b.symbol());
        let path: &Path = Path::new(&filename);
        let data: String = fs::read_to_string(path).expect("Unable to read file");

        SkfHandler {
            el_a,
            el_b,
            data_string: data,
        }
    }
}

/// Converts a line into a list of column values respecting the format conventions used
/// in the Slater-Koster files (.skf). Note: In these files, zero columns are not written!
/// e.g. 4*0.0 has to be replaced by four columns with zeros "0.0 0.0 0.0 0.0".
pub fn process_slako_line(line: &str) -> Vec<f64> {
    let line: String = line.replace(",", " ");
    let new_line: Vec<&str> = line.split(' ').collect();
    let mut float_vec: Vec<f64> = Vec::new();
    for string in new_line {
        if string.contains('*') {
            let temp: Vec<&str> = string.split('*').collect();
            let value: f64 = temp[1].trim().parse::<f64>().unwrap();
            for _ in 0..temp[0].trim().parse::<usize>().unwrap() {
                float_vec.push(value);
            }
        } else if !string.is_empty() && !string.contains('\t') {
            let value: f64 = string.trim().parse::<f64>().unwrap();
            float_vec.push(value);
        }
    }
    float_vec
}

pub fn get_tau_2_index(tuple: (u8, i32, u8, i32)) -> u8 {
    let value: u8 = match tuple {
        (0, 0, 0, 0) => 0,
        (0, 0, 1, 0) => 2,
        (0, 0, 2, 0) => 3,
        (1, 0, 0, 0) => 4,
        (1, -1, 1, -1) => 5,
        (1, 0, 1, 0) => 6,
        (1, 1, 1, 1) => 5,
        (1, -1, 2, -1) => 7,
        (1, 0, 2, 0) => 8,
        (1, 1, 2, 1) => 7,
        (2, 0, 0, 0) => 9,
        (2, -1, 1, -1) => 10,
        (2, 0, 1, 0) => 11,
        (2, 1, 1, 1) => 10,
        (2, -2, 2, -2) => 12,
        (2, -1, 2, -1) => 13,
        (2, 0, 2, 0) => 14,
        (2, 1, 2, 1) => 13,
        (2, 2, 2, 2) => 12,
        _ => panic!("false combination for tau_2_index!"),
    };
    value
}

pub fn get_index_to_symbol() -> HashMap<u8, String> {
    HashMap::from([
        (0, "ss_sigma".to_string()),
        (2, "ss_sigma".to_string()),
        (3, "sp_sigma".to_string()),
        (4, "sd_sigma".to_string()),
        (5, "ps_sigma".to_string()),
        (6, "pp_pi".to_string()),
        (7, "pp_sigma".to_string()),
        (8, "pd_pi".to_string()),
        (9, "pd_sigma".to_string()),
        (10, "ds_sigma".to_string()),
        (11, "dp_pi".to_string()),
        (12, "dp_sigma".to_string()),
        (13, "dd_delta".to_string()),
        (14, "dd_pi".to_string()),
    ])
}
